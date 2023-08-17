import logging
from abc import abstractmethod
from enum import Enum
from typing import List

from btt.exp_manager import MonitorRuleConfig
from btt.utils import if_nan_or_inf


class AssessResult(Enum):
    Good = 0
    Bad = 1


class AssessorIndicatorBase:
    # #     indicator_class_name_list = ['AgvIndicator', 'EagIndicator', 'ErgIndicator', 'PlcIndicator', 'LarIndicator',
    # #                                  'UlcIndicator', 'NmgIndicator']
    monitor_config: List[MonitorRuleConfig] = None  # shared ctx?

    def __init__(self, d_args):
        self.indicator_name = d_args["indicator_name"]
        self.logger = logging.getLogger(self.indicator_name)

    @abstractmethod
    def assess_trial(self, trial_id, trial_id_result_dict_list, trial_id_params_dict):
        raise NotImplementedError


class StatisticIndicatorBase(AssessorIndicatorBase):

    def __init__(self, d_args):
        super().__init__(d_args)

    def get_module_statistic_array(self, class_name, metric_prefix, metric_suffix, result_dict):
        column_idx = None
        for rule_config in self.monitor_config:
            if rule_config.name == class_name and rule_config.init_args["metric_prefix"] == metric_prefix:
                metric_suffix_list = rule_config.init_args["metric_suffix_list"]
                column_idx = metric_suffix_list.index(metric_suffix)
                if column_idx == -1:
                    raise ValueError("metric_suffix:{} not found".format(metric_suffix, metric_suffix_list))
        if column_idx is None:
            raise ValueError("rule_class:{} metric_prefix:{} not found".format(class_name, metric_prefix))
        array = None
        for rule_name, rule_result in result_dict.items():
            rule_class_name = self.monitor_config[rule_name].class_name
            if rule_class_name == class_name:
                array = rule_result[rule_name][:, column_idx]
        if array is None:
            raise ValueError("rule_class:{} not found".format(class_name))
        return array

    @abstractmethod
    def assess_trial(self, trial_id, trial_id_result_dict_list, trial_id_params_dict):
        raise NotImplementedError


class AgvIndicator(AssessorIndicatorBase):
    def __init__(self, d_args):
        super().__init__(d_args)

    def assess_trial(self, trial_id, trial_id_result_dict_list, trial_id_params_dict):
        # 可以测 int/float/tensor

        result_dict = trial_id_result_dict_list[trial_id][-1]
        for rule_name, rule_result in result_dict.items():
            rule_class_name = self.monitor_config[rule_name].class_name
            if rule_class_name == "ModePeriodMonitorRule":  # 我制定的我知道 train/val的acc/loss
                if if_nan_or_inf(result_dict[rule_name]):  # list(val)
                    return AssessResult.Bad
            elif rule_class_name in ["WeightStatisticsMonitorRule", "FeatureStatisticsMonitorRule"]:
                if if_nan_or_inf(rule_result[rule_name]):  # np_array(val)
                    return AssessResult.Bad
        return AssessResult.Good


class EagIndicator(StatisticIndicatorBase):
    def __init__(self, d_args):
        super().__init__(d_args)
        self.eag_t1 = 10 * 3

    def assess_trial(self, trial_id, trial_id_result_dict_list, trial_id_params_dict):
        # result_dict
        # {"rule_name1": {class_name: class_name1, result: result1},
        # "rule_name2": {class_name: class_name2, result: result2}}
        result_dict = trial_id_result_dict_list[trial_id][-1]
        array = self.get_module_statistic_array("WeightStatisticsMonitorRule", "weight_grad_abs", "avg", result_dict)

        if if_nan_or_inf(array):
            return AssessResult.Bad  # 直接爆炸（独立规则
        if (array == 0).any():
            return AssessResult.Good  # 有0不考虑梯度爆炸，考虑梯度消失
        val_list = list(array)
        for i in range(len(val_list) - 1):
            if val_list[i] / val_list[i + 1] > self.eag_t1:
                return AssessResult.Bad
        return AssessResult.Good


class ErgIndicator(StatisticIndicatorBase):
    def __init__(self, d_args):
        super().__init__(d_args)
        self.erg_t1 = 1 / 10 * 3

    def assess_trial(self, trial_id, trial_id_result_dict_list, trial_id_params_dict):
        result_dict = trial_id_result_dict_list[trial_id][-1]
        array = self.get_module_statistic_array("WeightStatisticsMonitorRule", "weight_grad_abs", "avg", result_dict)

        if if_nan_or_inf(array):
            return AssessResult.Good  # 直接爆炸,无法计算，不管
        if (array == 0).any():
            return AssessResult.Bad  # 有0直接梯度消失
        val_list = list(array)
        for i in range(len(val_list) - 1):
            if val_list[i] / val_list[i + 1] < self.erg_t1:
                return AssessResult.Bad
        return AssessResult.Good

class PlcIndicator(AssessorIndicatorBase):
    def __init__(self, d_args):
        super().__init__(d_args)
        self.plc_t1 = 1 / 10 * 3
        self.plc_t2 = 1 / 10 * 3

    def assess_trial(self, trial_id, trial_id_result_dict_list, trial_id_params_dict):
        result_dict = trial_id_result_dict_list[trial_id][-1]
        array = self.get_module_statistic_array("ModePeriodMonitorRule", "train_loss", "avg", result_dict)

        if if_nan_or_inf(array):
            return AssessResult.Good  # 直接爆炸,无法计算，不管
        val_list = list(array)
        for i in range(len(val_list) - 1):
            if val_list[i] / val_list[i + 1] < self.plc_t1:
                return AssessResult.Bad
        for i in range(len(val_list) - 1):
            if val_list[i] / val_list[i + 1] > self.plc_t2:
                return AssessResult.Bad
        return AssessResult.Good
