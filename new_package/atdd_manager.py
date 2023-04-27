import logging
import os

import nni

from atdd_messenger import ATDDMessenger
from atdd_monitor import ATDDMonitor
from atdd_utils import set_seed

logger = logging.getLogger(__name__)


class ATDDManager:
    def __init__(self, seed=None):
        set_seed(seed, "manager", logger)
        self.seed = seed

        self.advisor_config = ATDDMessenger().read_advisor_config()
        self.shared_config = None
        self.monitor_config = None
        self.assessor_config = None
        self.init_configs()

        self.monitor = ATDDMonitor(**self.monitor_config) if self.monitor_config is not None else None

        self.raw_mode = False if self.shared_config is not None and self.monitor_config is not None else True  # no inspect/assess maybe tuner

    def init_configs(self):
        if self.advisor_config is None:
            return
        self.shared_config = self.advisor_config["shared"] \
            if "shared" in self.advisor_config else None
        self.monitor_config = self.advisor_config["monitor"]["classArgs"] \
            if "monitor" in self.advisor_config else None
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

    def calculate_after_training(self):
        if self.monitor_config is not None:
            self.monitor.calculate_after_training()

    def collect_after_validating(self, *args):
        if self.monitor_config is not None:
            self.monitor.collect_after_validating(*args)

    def collect_after_testing(self, *args):
        if self.monitor_config is not None:
            self.monitor.collect_after_testing(*args)

    def init_basic(self, *args):
        if self.monitor_config is not None:
            self.monitor.init_basic(*args)

    def init_cond(self, *args):
        if self.monitor_config is not None:
            self.monitor.init_cond(*args)

    def refresh_before_epoch_start(self):
        if self.monitor_config is not None:
            self.monitor.refresh_before_epoch_start()

    def report_intermediate_result(self, rd=None, writer=None):
        d = {}
        if self.monitor_config is not None:
            d1 = self.monitor.get_intermediate_dict()
            ATDDMessenger().write_monitor_info(d1)
            d.update(d1)
        if self.raw_mode is True:
            d = self.get_raw_dict(rd)
        logger.info(" ".join(["intermediate_result_dict:", str(d)]))
        nni.report_intermediate_result(d)  # assessor _metric
        return d
        # manager考虑从assessor和inspector收回信息？？？

    def report_final_result(self, rd=None, writer=None):
        d = {}
        if self.monitor_config is not None:
            d1 = self.monitor.get_final_dict()
            ATDDMessenger().write_monitor_info(d1)
            d.update(d1)
        if self.assessor_config is not None:
            d3 = ATDDMessenger().read_assessor_info()
            while d3 is None:
                d3 = ATDDMessenger().read_assessor_info()
                os.system("sleep 1")
            d.update(d3)
        if self.raw_mode is True:
            d = self.get_raw_dict(rd)
        logger.info(" ".join(["final_result_dict:", str(d)]))
        nni.report_final_result(d)  # tuner symptom
        return d

    def if_atdd_send_stop(self):
        if self.assessor_config is not None:
            early_stop = ATDDMessenger().if_atdd_assessor_send_stop()
            if early_stop:
                info_dict = ATDDMessenger().read_assessor_info()
                logger.info(" ".join(["assessor_info_dict ", str(info_dict)]))
            return early_stop
        return False
