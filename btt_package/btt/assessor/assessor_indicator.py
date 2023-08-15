import logging
from abc import abstractmethod
from enum import Enum

import numpy as np


class AssessResult(Enum):
    Good = 0
    Bad = 1


class AssessorIndicatorBase:
    # #     indicator_class_name_list = ['AgvIndicator', 'EagIndicator', 'ErgIndicator', 'PlcIndicator', 'LarIndicator',
    # #                                  'UlcIndicator', 'NmgIndicator']
    def __init__(self, d_args):
        self.indicator_name = d_args["indicator_name"]
        self.logger = logging.getLogger(self.indicator_name)

    @abstractmethod
    def assess_trial(self, param_id, result_dict_list, param_id_result_dict_list):
        raise NotImplementedError


class AgvIndicator(AssessorIndicatorBase):
    def __init__(self, d_args):
        super().__init__(d_args)

    def assess_trial(self, param_id, result_dict_list, param_id_result_dict_list):
        def if_nan_or_inf(x):
            return np.isnan(x) or np.isinf(x)  # 可以测 int/float/tensor

        result_dict = result_dict_list[-1]
        key_list1 = ["train_acc", "val_acc", "train_loss", "val_loss"]
        key_list2 = ["weight_val", "weight_grad", "feature_val_in", "feature_grad_out"]
        for rule_name in result_dict.keys():
            if rule_name in key_list1:
                if if_nan_or_inf(result_dict[rule_name]):
                    return AssessResult.Bad
            elif rule_name in key_list2:
                array = result_dict[rule_name]
                if np.isnan(array).any() or np.isinf(array).any():
                    return AssessResult.Bad
        return AssessResult.Good

class EagIndicator(AssessorIndicatorBase):
    def __init__(self, d_args):
        super().__init__(d_args)

    def assess_trial(self, param_id, result_dict_list, param_id_result_dict_list):
        result_dict = result_dict_list[-1]
        if result_dict["val_acc"] < 0.5:
            return AssessResult.Bad
        return AssessResult.Good
