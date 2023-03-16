import logging

import nni

from atdd_inspector import ATDDInspector
from atdd_messenger import ATDDMessenger
from atdd_monitor import ATDDMonitor
from atdd_utils import set_seed

logger = logging.getLogger(__name__)


class ATDDManager:
    def __init__(self, seed=None):
        set_seed(seed, "manager", logger)

        self.advisor_config = ATDDMessenger().read_advisor_config()
        self.shared_config = None
        self.monitor_config = None
        self.inspector_config = None
        self.assessor_config = None
        self.init_configs()

        self.monitor = ATDDMonitor(**self.monitor_config) if self.monitor_config is not None else None
        self.inspector = ATDDInspector(**self.inspector_config) if self.inspector_config is not None else None
        self.model_num = self.advisor_config["shared"]["model_num"] \
            if self.shared_config is not None else None

        self.raw_mode = False if self.shared_config is not None else True
        # self.raw_dict = None ...

        self.assessor_stop = False
        self.inspector_stop = False

    def init_configs(self):
        if self.advisor_config is None:
            return
        self.shared_config = self.advisor_config["shared"] \
            if "shared" in self.advisor_config else None
        self.monitor_config = self.advisor_config["monitor"]["classArgs"] \
            if "monitor" in self.advisor_config else None
        self.inspector_config = self.advisor_config["inspector"]["classArgs"] \
            if "inspector" in self.advisor_config else None
        self.assessor_config = self.advisor_config["assessor"]["classArgs"] \
            if "assessor" in self.advisor_config else None

    def get_raw_dict(self, result_dict):
        if type(result_dict):
            return result_dict
        elif type(result_dict) is int or type(result_dict) is float:
            return {"default": result_dict}
        else:
            return {"default": 0}

    def collect_in_training(self, *args):
        if self.monitor_config is not None:
            self.monitor.collect_in_training(*args)

    def collect_after_training(self, *args):
        if self.monitor_config is not None:
            self.monitor.collect_after_training(*args)
        # if self.raw_mode is True:
        #     self.raw_update()

    def calculate_metrics_after_training(self):
        if self.monitor_config is not None:
            self.monitor.calculate_metrics_after_training()

    def collect_after_validating(self, *args):
        if self.monitor_config is not None:
            self.monitor.collect_after_validating(*args)

    def collect_after_testing(self, *args):
        if self.monitor_config is not None:
            self.monitor.collect_after_testing(*args)

    def init_module_basic(self, *args):
        if self.monitor_config is not None:
            self.monitor.init_module_basic(*args)

    def init_cond(self, *args):
        if self.monitor_config is not None:
            self.monitor.init_cond(*args)

    def refresh_before_epoch_start(self):
        if self.monitor_config is not None:
            self.monitor.refresh_before_epoch_start()

    def report_intermediate_result(self, rd=None):
        d = {}
        if self.monitor_config is not None:
            d1 = self.monitor.get_intermediate_dict()
            ATDDMessenger().write_monitor_info(d1)
            d.update(d1)
        if self.inspector_config is not None:
            d2 = self.inspector.load_and_get_dict()
            ATDDMessenger().write_inspector_info(d2)
            d.update(d2)
        if self.raw_mode is True:
            d = self.get_raw_dict(rd)
        logger.info(" ".join(["intermediate_result_dict:", str(d)]))
        nni.report_intermediate_result(d)  # assessor _metric
        return d
        # manager考虑从assessor和inspector收回信息？？？

    def report_final_result(self, rd=None):
        d = {}
        if self.monitor_config is not None:
            d1 = self.monitor.get_final_dict()
            ATDDMessenger().write_monitor_info(d1)
            d.update(d1)
        if self.inspector_config is not None:
            d2 = ATDDMessenger().read_inspector_info()
            d.update(d2)
        if self.assessor_config is not None:
            d3 = ATDDMessenger().read_assessor_info()
            d.update(d3)
        d.update({"assessor_stop": self.assessor_stop})
        d.update({"inspector_stop": self.inspector_stop})
        if self.raw_mode is True:
            d = self.get_raw_dict(rd)
        logger.info(" ".join(["final_result_dict:", str(d)]))
        nni.report_final_result(d)  # tuner symptom
        return d

    def if_atdd_inspector_send_stop(self):
        if self.inspector_config is not None:
            info_dict = ATDDMessenger().read_inspector_info()
            if info_dict is None:
                return False
            for k, v in info_dict.items():
                if "_symptom" in k and v is not None:
                    print("inspector_info_dict ", info_dict)
                    self.inspector_stop = True
                    return True
            return False
        else:
            return False

    def if_atdd_assessor_send_stop(self):
        if self.assessor_config is not None:
            info_dict = ATDDMessenger().read_assessor_info()
            if info_dict is None:
                return False
            for k, v in info_dict.items():
                if "cmp_" in k and v is not None:
                    print("assessor_info_dict ", info_dict)
                    self.assessor_stop = True
                    return True
            return False
        else:
            return False

    def if_atdd_send_stop(self):
        return self.if_atdd_inspector_send_stop() or self.if_atdd_assessor_send_stop()
