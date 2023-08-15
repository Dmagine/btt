import importlib
import logging
import os
import sys
import threading

import nni

from btt_messenger import BttMessenger
from monitor_rule import OnceMonitorRuleBase, PeriodMonitorRuleBase
from utils import ObtainMode

logger = logging.getLogger(__name__)


class BttMonitor:
    rule_config = None  # rule_name: file_path class_name init_args
    rule_name_list = None
    rule_instance_dict = {}  # sub_monitor_name: sub_monitor_object

    intermediate_expect_idx = 0
    intermediate_actual_idx = 0
    intermediate_rule_name_list = []
    intermediate_thread = None
    intermediate_thread_stop = False
    intermediate_lock = threading.Lock()
    intermediate_default_rule_name = None

    final_report_rule_name_list = []
    final_default_rule_name = None

    def __init__(self, rule_config):
        logger.debug("monitor hello!")
        self.rule_config = rule_config
        self.rule_name_list = list(self.rule_config.keys())
        self.init_rule_instance()
        self.init_intermediate_report()
        self.init_final_report()

    def init_rule_instance(self):
        for rule_name, rule_info in self.rule_config.items():
            # if mode == "once":
            #     instance = OnceMonitorRuleBase({"rule_name": rule_name})
            # elif mode == "period":
            #     instance = PeriodMonitorRuleBase({"rule_name": rule_name})
            code_dir = rule_info["code_dir"]
            module_name = rule_info["module_name"]
            class_name = rule_info["class_name"]
            init_args = rule_info["init_args"] if "init_args" in rule_info else {}
            init_args["rule_name"] = rule_name

            sys.path.append(code_dir)
            class_module = importlib.import_module(module_name)
            class_constructor = getattr(class_module, class_name)
            instance = class_constructor(init_args)
            self.rule_instance_dict[rule_name] = instance

            if "intermediate_default" in rule_info and rule_info["intermediate_default"]:
                self.intermediate_default_rule_name = rule_name
            if "final_default" in rule_info and rule_info["final_default"]:
                self.final_default_rule_name = rule_name

    def record_metric(self, d_args):
        logger.debug(" ".join(["record_metric:", str(d_args)]))
        for rule_instance in self.rule_instance_dict.values():
            rule_instance.record_metric(d_args)

    def obtain_metric(self, rule_name, d_args):
        # mode / result_idx
        logger.debug(" ".join(["obtain_metric:", rule_name, str(d_args)]))
        if rule_name not in self.rule_instance_dict:
            raise ValueError("metric_name {} should exist in rule_instance_dict".format(rule_name))
        r = self.rule_instance_dict[rule_name].obtain_metric(d_args)
        return r

    def _report_intermediate_result(self):
        while True:
            with self.intermediate_lock:
                if self.intermediate_expect_idx > self.intermediate_actual_idx:
                    d = {}  # 每次尝试汇报都要清空
                    for rule_name in self.intermediate_rule_name_list:
                        d_args = {"result_idx": self.intermediate_actual_idx, "mode": ObtainMode.IdxWait}
                        tmp_d = self.obtain_metric(rule_name, d_args)
                        if tmp_d is not None:
                            d[rule_name] = tmp_d
                        else:
                            logger.debug(" ".join(["_report_intermediate_result none:",
                                                   rule_name, str(self.intermediate_actual_idx),
                                                   str(self.intermediate_expect_idx)]))

                    if list(d.keys()) == self.intermediate_rule_name_list:  # ok
                        d["default"] = d[self.intermediate_default_rule_name]
                        BttMessenger().add_intermediate_monitor_result(d, self.intermediate_actual_idx)
                        nni.report_intermediate_result(d)
                        logger.info(" ".join(["intermediate_report:", str(self.intermediate_actual_idx)]))
                        self.intermediate_actual_idx += 1
                if self.intermediate_thread_stop:
                    break
            os.system("sleep 1")

    def init_intermediate_report(self):
        # 期望 不阻塞 难实现 多个sub难以协调
        logger.debug("init_intermediate_report:")
        for rule_name, rule_info in self.rule_config.items():
            if rule_info["intermediate_report"] is True:
                self.intermediate_rule_name_list.append(rule_name)
        self.intermediate_thread = threading.Thread(target=self._report_intermediate_result)
        # self.intermediate_thread.Daemon = True #....
        self.intermediate_thread.start()

    def init_final_report(self):
        logger.debug("init_final_report:")
        for rule_name, rule_info in self.rule_config.items():
            if rule_info["final_report"] is True:
                self.final_report_rule_name_list.append(rule_name)

    def report_final_result(self):
        logger.debug("report_final_result begin:")
        # final 和 intermediate 的 obtain 需要按照顺序

        while True:  # 等待中间结果汇报全部完成
            with self.intermediate_lock:
                if self.intermediate_expect_idx == self.intermediate_actual_idx:
                    self.intermediate_thread_stop = True
                    break
            os.system("sleep 1")
        idx1, idx2 = self.intermediate_actual_idx, self.intermediate_expect_idx
        logger.debug(" ".join(["report_final_result wait:", str(idx1), str(idx2)]))
        self.intermediate_thread.join()

        d = {}  # 等待rule calc全部完成
        for rule_name in self.rule_name_list:
            logger.debug(" ".join(["report_final_result wait:", rule_name]))
            d_args = {"result_idx": -1, "mode": ObtainMode.AllWait}
            metric_value = self.obtain_metric(rule_name, d_args)
            if rule_name in self.final_report_rule_name_list:
                d.update({rule_name: metric_value})
        logger.info("final_report report: ")

        d["default"] = d[self.final_default_rule_name]
        BttMessenger().add_final_monitor_result(d)
        nni.report_final_result(d)


if __name__ == '__main__':
    pass
