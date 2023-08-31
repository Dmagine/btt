import logging
from abc import abstractmethod
from enum import Enum

import numpy as np
from btt.utils import if_nan_or_inf


class AssessResult(Enum):
    Good = 0
    Bad = 1


class AssessorIndicatorBase:
    # #     indicator_class_name_list = ['AgvIndicator', 'EagIndicator', 'ErgIndicator', 'PlcIndicator', 'LarIndicator',
    # #                                  'UlcIndicator', 'NmgIndicator']
    # monitor_config: List[MonitorRuleConfig] = None  # shared ctx?

    def __init__(self, indicator_name):
        self.indicator_name = indicator_name
        # self.monitor_config = d_args["monitor_config"]
        self.logger = logging.getLogger(self.indicator_name)
        self.id_initial_dict = None
        self.id_interm_list_dict = None
        self.id_params_dict = None

    def update_initial

    def before_assess(self, id_initial_dict_dict, id_intermediate_dict_list_dict, id_parameters_dict):
        # 复杂数据传到这
        self.id_initial_dict = id_initial_dict_dict
        self.id_interm_list_dict = id_intermediate_dict_list_dict
        self.id_params_dict = id_parameters_dict
        # self.initial_dict = id_initial_dict_dict[self.indicator_name]
        # self.intermediate_dict_list = id_intermediate_dict_list_dict[self.indicator_name]
        # self.parameters = id_parameters_dict[self.indicator_name]
        pass

    @abstractmethod
    def assess_trial(self, initial_dict, intermediate_dict_list):
        raise NotImplementedError


class WindowIndicatorBase(AssessorIndicatorBase):

    def __init__(self, d_args):
        super().__init__(d_args)
        self.window_ratio = 0.25 if "window_ratio" not in d_args else d_args["window_ratio"]

    def get_window_size(self, initial_dict, window_size_ratio, default=5):
        window_size = default
        for rule_name, rule_result in initial_dict.items():
            rule_class_name = self.monitor_config[rule_name].class_name
            if rule_class_name == "CommonInfoMonitorRule":  # initial report / Common
                max_nb_epoch = rule_result["max_nb_epoch"]
                window_size = max(window_size, int(round(window_size_ratio * max_nb_epoch)))
                break
        return window_size

    @abstractmethod
    def assess_trial(self, initial_dict, intermediate_dict_list):
        raise NotImplementedError


class LossIndicatorBase(AssessorIndicatorBase):
    def __init__(self, d_args):
        super().__init__(d_args)
        self.loss_token_list = ["loss", "mse", "mae", "rmse"] \
            if "loss_token_name_list" not in d_args else d_args["loss_token_name_list"]
        self.min_loss_list_len = 3 \
            if "min_loss_list_len" not in d_args else d_args["min_loss_list_len"]
        # minimize

    def if_loss_criterion(self, rule_name, loss_list):
        if type(loss_list) is not list:  # 默认要求
            return False
        rule_class_name = self.monitor_config[rule_name].class_name
        if rule_class_name != "ModePeriodMonitorRule":
            return False

        key = self.monitor_config[rule_name]["init_args"]["key"]
        return any([token in word for token in self.loss_token_list for word in [rule_name, key]])

    @abstractmethod
    def assess_trial(self, initial_dict, intermediate_dict_list):
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
                array = rule_result[rule_name]["array"][:, column_idx]
        if array is None:
            raise ValueError("rule_class:{} not found".format(class_name))
        return array

    @abstractmethod
    def assess_trial(self, initial_dict, intermediate_dict_list):
        raise NotImplementedError


class AgvIndicator(AssessorIndicatorBase):
    def __init__(self, d_args):
        super().__init__(d_args)

    def assess_trial(self, initial_dict, intermediate_dict_list):
        # 可以测 int/float/tensor

        result_dict = intermediate_dict_list[-1]
        for rule_name, rule_result in result_dict.items():
            rule_class_name = self.monitor_config[rule_name].class_name
            if rule_class_name == "ModePeriodMonitorRule":  # 我制定的我知道 train/val的acc/loss
                if if_nan_or_inf(result_dict[rule_name]):  # list(val)
                    return AssessResult.Bad
            elif rule_class_name in ["WeightStatisticsMonitorRule", "FeatureStatisticsMonitorRule"]:
                if if_nan_or_inf(rule_result[rule_name]["array"]):  # np_array(val)
                    return AssessResult.Bad
        return AssessResult.Good


class EagIndicator(StatisticIndicatorBase):
    def __init__(self, d_args):
        super().__init__(d_args)
        self.T1 = 10 ** 3

    def assess_trial(self, initial_dict, intermediate_dict_list):
        # result_dict
        # {"rule_name1": {class_name: class_name1, result: result1},
        # "rule_name2": {class_name: class_name2, result: result2}}
        result_dict = intermediate_dict_list[-1]
        array = self.get_module_statistic_array("WeightStatisticsMonitorRule", "weight_grad_abs", "avg", result_dict)

        if if_nan_or_inf(array):
            return AssessResult.Bad  # 直接爆炸（独立规则
        if (array == 0).any():
            return AssessResult.Good  # 有0不考虑梯度爆炸，考虑梯度消失
        val_list = list(array)
        for i in range(len(val_list) - 1):
            if val_list[i] / val_list[i + 1] > self.T1:
                return AssessResult.Bad
        return AssessResult.Good


class ErgIndicator(StatisticIndicatorBase):
    def __init__(self, d_args):
        super().__init__(d_args)
        self.T1 = 1 / 10 ** 3

    def assess_trial(self, initial_dict, intermediate_dict_list):
        result_dict = intermediate_dict_list[-1]
        array = self.get_module_statistic_array("WeightStatisticsMonitorRule", "weight_grad_abs", "avg", result_dict)

        if if_nan_or_inf(array):
            return AssessResult.Good  # 直接爆炸,无法计算，不管
        if (array == 0).any():
            return AssessResult.Bad  # 有0直接梯度消失
        val_list = list(array)
        for i in range(len(val_list) - 1):
            if val_list[i] / val_list[i + 1] < self.T1:
                return AssessResult.Bad
        return AssessResult.Good


class PlcIndicator(LossIndicatorBase):
    def __init__(self, d_args):
        super().__init__(d_args)
        self.T1 = 1 / 10

    def assess_trial(self, initial_dict, intermediate_dict_list):
        result_dict = intermediate_dict_list[-1]
        for rule_name, loss_list in result_dict.items():
            if self.if_loss_criterion(rule_name, loss_list):
                continue
            if if_nan_or_inf(result_dict[rule_name]):
                return AssessResult.Bad
            if len(loss_list) < self.min_loss_list_len:
                return AssessResult.Good
            stop_flag = (sum([abs(loss_list[i + 1] - loss_list[i]) for i in range(len(loss_list) - 1)]) / (
                    len(loss_list) - 1)) / loss_list[0] < self.T1
            if stop_flag:
                return AssessResult.Bad
        return AssessResult.Good


class UlcIndicator(WindowIndicatorBase, LossIndicatorBase):
    def __init__(self, d_args):
        super().__init__(d_args)
        self.T1 = 1 / 10
        self.T2 = 1 / 10 ** 2

    def assess_trial(self, initial_dict, intermediate_dict_list):
        result_dict = intermediate_dict_list[-1]
        window_size = self.get_window_size(initial_dict, self.window_ratio)

        for rule_name, loss_list in result_dict.items():
            if self.if_loss_criterion(rule_name, loss_list):
                continue
            if if_nan_or_inf(result_dict[rule_name]):
                return AssessResult.Bad
            if len(loss_list) < self.min_loss_list_len:
                return AssessResult.Good
            sub_list = loss_list[-window_size:]
            x = np.arange(len(sub_list))
            line = np.polyfit(x, sub_list, 1)
            mae = np.mean(np.abs(sub_list - np.polyval(line, x)))
            if mae > np.mean(sub_list) * self.T2:
                return AssessResult.Bad
        return AssessResult.Good


class NmgIndicator(WindowIndicatorBase, LossIndicatorBase):
    def __init__(self, d_args):
        super().__init__(d_args)

    def assess_trial(self, initial_dict, intermediate_dict_list):
        result_dict = intermediate_dict_list[-1]
        window_size = self.get_window_size(initial_dict, self.window_ratio)
        for rule_name, loss_list in result_dict.items():
            if self.if_loss_criterion(rule_name, loss_list):
                continue
            if if_nan_or_inf(result_dict[rule_name]):
                return AssessResult.Bad
            if len(loss_list) < self.min_loss_list_len:
                return AssessResult.Good
            min_loss = np.min(loss_list)
            sub_list = loss_list[-window_size:]
            min_loss_sub = np.min(sub_list)
            if min_loss_sub > min_loss:
                return AssessResult.Bad
        return AssessResult.Good
