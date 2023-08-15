import logging

import nni

from btt_messenger import BttMessenger
from btt_package.btt.monitor.monitor import BttMonitor
from utils import set_seed

logger = logging.getLogger(__name__)


class BttTrialManager:
    # monitor: resource
    # tuner: 1 get_trial_parameters 2 report_trial_final_result (资源不相关)
    # assessor: 1 get_trial_resource_quota 2 report_intermediate_result (资源强相关)
    advisor_config = None
    monitor_config = None
    assessor_config = None

    monitor = None

    def __init__(self, seed=None):
        set_seed(seed, "btt_trial_manager", logger)
        self.seed = seed

        self.advisor_config = BttMessenger().read_advisor_config()
        self.shared_config = self.advisor_config["shared"] \
            if self.advisor_config is not None and "shared" in self.advisor_config else None
        self.monitor_config = self.advisor_config["monitor"]["classArgs"] \
            if self.advisor_config is not None and "monitor" in self.advisor_config else None
        self.assessor_config = self.advisor_config["assessor"]["classArgs"] \
            if self.advisor_config is not None and "assessor" in self.advisor_config else None
        self.monitor = BttMonitor(**self.monitor_config) if self.monitor_config is not None else None

    def record_metric(self, d_args):
        if self.monitor_config is not None:
            # for rule_name, d_args in d2.items():
            #     self.monitor.record_metric(rule_name, d_args)
            self.monitor.record_metric(d_args)

    def obtain_metric(self, d_args):
        # metric_name or metric_name_list
        rule_name = d_args["rule_name"]
        if self.monitor_config is not None:
            if type(rule_name) is list:
                metric_name_list = rule_name
                d = {}
                for rule_name in metric_name_list:
                    d_args_ = d_args.copy()
                    d[rule_name] = self.monitor.obtain_metric(rule_name, d_args)
            else:
                return self.monitor.obtain_metric(rule_name, d_args)

    def get_experiment_id(self):
        return nni.get_experiment_id()

    def get_trial_id(self):
        return nni.get_trial_id()

    def get_trial_parameters(self):
        return nni.get_next_parameter()

    def update_resource_params(self, resource_params):
        logger.debug("resource_params: {}".format(resource_params))
        if self.assessor_config is not None:
            d = BttMessenger().read_resource_params(resource_params)
            if d is not None:
                resource_params.update(d)
            return resource_params
        else:
            return resource_params

    def report_intermediate_result(self):
        if self.monitor_config is not None:
            self.monitor.intermediate_expect_idx += 1

    def report_final_result(self):
        if self.monitor_config is not None:
            self.monitor.report_final_result()
