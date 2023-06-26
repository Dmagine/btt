import logging

import nni

from atdd_messenger import ATDDMessenger
from atdd_utils import set_seed
from btt_monitor import BttMonitor, MonitorCategory, monitor_str_2_class

logger = logging.getLogger(__name__)


class BttTrialManager:
    advisor_config = None
    shared_config = None
    monitor_config = None
    assessor_config = None
    monitor = None

    # current_metric_name = None  # -> record metric
    # metric_dict = None  # -> get metric
    # resource_params = None  # -> get resource

    def __init__(self, seed=None):
        set_seed(seed, "btt_trial_manager", logger)
        self.seed = seed

        self.advisor_config = ATDDMessenger().read_advisor_config()
        self.shared_config = self.advisor_config["shared"] \
            if self.advisor_config is not None and "shared" in self.advisor_config else None
        self.monitor_config = self.advisor_config["monitor"]["classArgs"] \
            if self.advisor_config is not None and "monitor" in self.advisor_config else None
        self.assessor_config = self.advisor_config["assessor"]["classArgs"] \
            if self.advisor_config is not None and "assessor" in self.advisor_config else None
        self.monitor = BttMonitor(**self.monitor_config) if self.monitor_config is not None else None

    #     self.locks = threading.Lock()
    #     self.monitor_queue = queue.Queue()
    #     self.thread = threading.Thread(target=self._monitor_run)
    #     self.thread.start()
    #
    # def _monitor_run(self):
    #     while True:
    #         with self.lock1:
    #             k, v, c = self.monitor_queue.get()
    #             self.current_metric_name = k
    #             if c == MonitorCategory.RULE:
    #                 self.monitor.record_with_rule(k, v)  # metric_name, metric_value
    #             elif c == MonitorCategory.PERIOD:
    #                 self.monitor.record_periodically(k, v)  # metric_name, metric_value
    #             elif c == MonitorCategory.ONCE:
    #                 self.monitor.record_once(k, v)  # rule_name, kwargs
    #             elif c == MonitorCategory.GET:
    #                 self.metric_dict = self.monitor.get_metric(k)
    #             elif c == MonitorCategory.INTERMEDIATE:
    #                 self.monitor.report_intermediate_result(k)
    #             elif c == MonitorCategory.FINAL:
    #                 self.monitor.report_final_result(k)
    #             elif c == MonitorCategory.RESOURCE:
    #                 self.resource_params = self.monitor.get_resource_parameters(k, v)
    #             else:
    #                 raise ValueError("unknown category")

    def record_metric(self, d, category="once"):
        # block = True
        c = monitor_str_2_class(category)
        if self.monitor_config is not None:
            for k, v in d:
                if c == MonitorCategory.RULE:
                    self.monitor.record_with_rule(k, v)  # metric_name, metric_value
                elif c == MonitorCategory.PERIOD:
                    self.monitor.record_periodically(k, v)  # metric_name, metric_value
                elif c == MonitorCategory.ONCE:
                    self.monitor.record_once(k, v)  # rule_name, kwargs

                elif c == MonitorCategory.GET:
                    self.metric_dict = self.monitor.get_metric(k)
                elif c == MonitorCategory.INTERMEDIATE:
                    self.monitor.report_intermediate_result(k)
                elif c == MonitorCategory.FINAL:
                    self.monitor.report_final_result(k)
                elif c == MonitorCategory.RESOURCE:
                    self.resource_params = self.monitor.get_resource_parameters(k, v)
                else:
                    raise ValueError("unknown category")
        else:
            raise ValueError("monitor_config is None")

    def get_metric(self, metric_name):
        # block = True
        if self.monitor_config is not None:
            return self.monitor.get_metric(metric_name)

    def get_experiment_id(self):
        return nni.get_experiment_id()

    def get_trial_id(self):
        return nni.get_trial_id()

    def get_trial_parameters(self):
        return nni.get_next_parameter()

    def get_resource_parameters(self):
        if self.monitor_config is not None:
            return self.monitor.get_resource_parameters()

    def report_intermediate_report(self, metric):
        if self.monitor_config is not None:
            d1 = self.monitor.get_intermediate_result_dict()

    def report_final_report(self, metric):
        if

    def

# def report_intermediate_result(self, rd=None):
#     d = {}
#     if self.monitor_config is not None:
#         d1 = self.monitor.get_intermediate_dict()
#         ATDDMessenger().write_monitor_info(d1)
#         d.update(d1)
#     if self.raw_mode is True:
#         d = self.get_raw_dict(rd)
#     logger.info(" ".join(["intermediate_result_dict:", str(d)]))
#     nni.report_intermediate_result(d)  # assessor _metric
#     return d
#     # manager考虑从assessor和inspector收回信息？？？
#
# def report_final_result(self, rd=None):
#     d = {}
#     if self.monitor_config is not None:
#         d1 = self.monitor.get_final_dict()
#         ATDDMessenger().write_monitor_info(d1)
#         d.update(d1)
#     if self.assessor_config is not None:
#         d3 = ATDDMessenger().read_assessor_info()
#         while d3 is None:
#             d3 = ATDDMessenger().read_assessor_info()
#             os.system("sleep 1")
#         d.update(d3)
#     if self.raw_mode is True:
#         d = self.get_raw_dict(rd)
#     logger.info(" ".join(["final_result_dict:", str(d)]))
#     nni.report_final_result(d)  # tuner symptom
#     return d
#
# def if_send_stop(self):
#     if self.assessor_config is not None:
#         early_stop = ATDDMessenger().if_atdd_assessor_send_stop()
#         if early_stop:
#             info_dict = ATDDMessenger().read_assessor_info()
#             logger.info(" ".join(["assessor_info_dict ", str(info_dict)]))
#         return early_stop
#     return False
