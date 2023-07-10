import logging

import nni

from btt_messenger import BttMessenger
from btt_monitor import BttMonitor
from btt_utils import set_seed

logger = logging.getLogger(__name__)


class BttTrialManager:
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

    def record_metric(self, d1):
        d2 = d1
        # d2 = deepcopy(d1)  # deep copy -> no grad (grad) / id fail (feature) / slow!
        if self.monitor_config is not None:
            for rule_name, d_args in d2.items():
                # if "model" in d_args:
                #     if d1[rule_name]["model"].conv1.weight.grad is not None:
                #         print(d1[rule_name]["model"].conv1.weight.grad.shape)
                #         print(d2[rule_name]["model"].conv1.weight.grad.shape)
                #         exit()
                self.monitor.record_metric(rule_name, d_args)
        del d2

    def obtain_metric(self, metric_name, idx=-1, mode="idx_wait"):
        # metric_name or metric_name_list
        if self.monitor_config is not None:
            if type(metric_name) is list:
                d = {}
                for m in metric_name:
                    d[m] = self.monitor.obtain_metric(m, idx, mode=mode)
            else:
                return self.monitor.obtain_metric(metric_name, idx, mode=mode)

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
