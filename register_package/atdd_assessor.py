import logging

import numpy as np
from nni.assessor import Assessor, AssessResult

from atdd_messenger import ATDDMessenger
from atdd_utils import set_seed

logger = logging.getLogger(__name__)


class ATDDAssessor(Assessor):
    def __init__(self, shared, basic, compare, diagnose, seed=None):
        super().__init__()
        self.etr = MyAssessor(shared, basic, compare, diagnose, seed)

    def assess_trial(self, trial_id, result_dict_list):
        early_stop = self.etr.assess_trial(trial_id, result_dict_list)
        ATDDMessenger(trial_id).write_assessor_info(self.etr.info_dict)
        # return AssessResult.Bad if early_stop else AssessResult.Good
        return AssessResult.Good  # 用户侧自己听


class AssessRuleBase:
    def __init__(self):
        pass

    def assess_trial(self, trial_id, result_dict_list):
        raise NotImplementedError


class MyAssessor:
    epoch_idx = 0
    trial_id = 0
    result_dict_list = []
    result_dict = {}
    info_dict = {}
    has_nan_inf = False

    def __init__(self, shared, basic, compare, diagnose, seed=None):
        set_seed(seed, "assessor", logger)
        self.shared = shared
        self.basic = basic
        self.compare = compare
        self.diagnose = diagnose

    def assess_trial(self, trial_id, result_dict_list):
        self.trial_id = trial_id
        self.result_dict_list = result_dict_list
        self.result_dict = self.result_dict_list[-1]
        self.has_nan_inf = False

        self.info_dict = self.get_default_info_dict()
        self.receive_monitor_result()

        self.diagnose_symptom()
        return self.assess_trial_end()

    def receive_monitor_result(self):
        def get_metric_array(p, s):
            idx = self.metric_prefix_list.index(p) * len(self.metric_suffix_list) + self.metric_suffix_list.index(s)
            return self.module_metric_2da[:, idx].flatten()

        d = self.result_dict
        self.epoch_idx = self.result_dict["epoch_idx"]
        self.module_name_list = d["module_name_list"]
        self.module_nele_list = d["module_nele_list"]
        self.has_nan_inf_list = d["has_nan_inf_list"]
        self.module_metric_2da = d["module_metric_2da"]
        self.metric_prefix_list = d["metric_prefix_list"]
        self.metric_suffix_list = d["metric_suffix_list"]
        # self.weight_grad_rate0_1da = get_metric_array("weight_grad_rate", "rate0")
        # self.weight_grad_abs_avg_1da = get_metric_array("weight_grad_abs", "avg")

        if type(self.weight_grad_abs_avg_1da) is dict:  # reproduce
            self.weight_grad_abs_avg_1da = np.array(self.weight_grad_abs_avg_1da["__ndarray__"])
            self.weight_grad_rate0_1da = np.array(self.weight_grad_rate0_1da["__ndarray__"])
        # print(len(self.weight_grad_abs_avg_1da), len(self.weight_grad_rate0_1da),
        #       len(self.module_nele_list), len(self.module_name_list))
        # print(self.trial_id, self.module_name_list)
