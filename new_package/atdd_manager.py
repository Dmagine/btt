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

        self.raw_mode = False if self.shared_config is not None and self.monitor_config is not None else True
        logger.info(" ".join(["raw_mode:", str(self.raw_mode)]))
        self.raw_result_dict_list = None
        self.raw_intermediate_default_key = None

    def init_configs(self):
        if self.advisor_config is None:
            return
        self.shared_config = self.advisor_config["shared"] \
            if "shared" in self.advisor_config else None
        self.monitor_config = self.advisor_config["monitor"]["classArgs"] \
            if "monitor" in self.advisor_config else None
        self.assessor_config = self.advisor_config["assessor"]["classArgs"] \
            if "assessor" in self.advisor_config else None

    def add_intermediate_default_in_raw(self):
        # after_train: "train_data_train_loss", "val_data_train_loss"
        # after_val: "val_data_train_loss"
        # after_test: "test_data_mse", "test_data_mae",...
        result_dict = self.raw_result_dict_list[-1]  # 同一个对象 修改有效
        if type(result_dict) is dict:
            if self.raw_intermediate_default_key is not None:
                result_dict["default"] = result_dict[self.raw_intermediate_default_key]
            else:
                for k, v in result_dict.items():
                    if "val_data_train_loss" in k:
                        result_dict["default"] = v
                        self.raw_intermediate_default_key = k
                        break
            if "default" not in result_dict:
                raise ValueError("raw_result dict error")
        elif type(result_dict) in [int, float]:
            pass
        else:
            raise ValueError("raw_result type error")
        logger.debug(" ".join(["add_intermediate_default_in_raw:", str(result_dict)]))
        logger.debug(" ".join(["add_intermediate_default_in_raw:", str(self.raw_result_dict_list[-1])]))

    def update_final_default_in_raw(self):
        result_dict = self.raw_result_dict_list[-1]  # test_data_mse_best
        if type(result_dict) is dict:
            for k, v in result_dict.items():
                if "test_data_mse" == k:
                    result_dict["default"] = v
                elif "test_data" in k:
                    result_dict["default"] = v
                    break
            if "default" not in result_dict:
                raise ValueError("raw_result dict error")
        else:
            pass  # val无传入best

    def collect_in_training(self, *args):
        if self.monitor_config is not None:
            self.monitor.collect_in_training(*args)

    def collect_after_training(self, *args):
        if self.monitor_config is not None:
            self.monitor.collect_after_training(*args)
        if self.raw_mode:
            self.raw_result_dict_list[-1].update(args[1])  # 传递元祖 其一None 其二loss

    def calculate_after_training(self):
        if self.monitor_config is not None:
            self.monitor.calculate_after_training()

    def collect_after_validating(self, *args):
        if self.monitor_config is not None:
            self.monitor.collect_after_validating(*args)
        if self.raw_mode:
            self.raw_result_dict_list[-1].update(args[1])  # 传递元祖 其一None 其二loss

    def collect_after_testing(self, *args):
        if self.monitor_config is not None:
            self.monitor.collect_after_testing(*args)
        if self.raw_mode:
            self.raw_result_dict_list[-1].update(args[1])  # 传递元祖 其一None 其二loss

    def init_basic(self, *args):
        if self.monitor_config is not None:
            self.monitor.init_basic(*args)

    def init_cond(self, *args):
        if self.monitor_config is not None:
            self.monitor.init_cond(*args)

    def refresh_before_epoch_start(self):
        if self.monitor_config is not None:
            self.monitor.refresh_before_epoch_start()
        if self.raw_mode:
            # self.raw_result_dict_list
            self.raw_result_dict_list = [] if self.raw_result_dict_list is None else self.raw_result_dict_list
            self.raw_result_dict_list.append({})

    def report_intermediate_result(self):
        d = {}
        if self.monitor_config is not None:
            d1 = self.monitor.get_intermediate_dict()
            ATDDMessenger().write_monitor_info(d1)
            d.update(d1)
        if self.raw_mode is True:
            self.add_intermediate_default_in_raw()
            d = self.raw_result_dict_list[-1]
        logger.info(" ".join(["intermediate_result_dict:", str(d)]))
        nni.report_intermediate_result(d)  # assessor _metric
        return d

    def report_final_result(self):
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
            self.add_intermediate_default_in_raw()
            self.update_final_default_in_raw()
            d = self.raw_result_dict_list[-1]
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
