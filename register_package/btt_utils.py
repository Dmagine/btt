import os
import random
from enum import Enum

import numpy as np
import torch
from scipy.stats import stats


def set_seed(seed, msg="", logger=None):
    if seed is None:
        seed = random.randint(11, 111)
    s = msg + "_seed: " + str(seed)
    print(s)
    if logger is not None:
        logger.info(s)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_monitor_config(advisor_config):
    monitor_config = advisor_config["monitor"]["classArgs"] \
        if advisor_config is not None and "monitor" in advisor_config else None
    return monitor_config


def get_assessor_config(advisor_config):
    assessor_config = advisor_config["assessor"]["classArgs"] \
        if advisor_config is not None and "assessor" in advisor_config else None
    return assessor_config


def get_module_name_nele_dict(model, module_type_list):
    d = {}
    for (module_name, module) in model.named_modules():
        if type(module) not in module_type_list:
            continue
        for (param_name, param) in module.named_parameters():
            if "weight" not in param_name:
                continue
            d.update({module_name: param.nelement()})
            break  # 一个网络层只统计一次
    return d


def get_module_id_name_dict(model, module_type_list):
    d = {}
    for (module_name, module) in model.named_modules():
        if type(module) not in module_type_list:
            continue
        for (param_name, param) in module.named_parameters():
            if "weight" not in param_name:
                continue
            d.update({id(module): module_name})
            break  # 一个网络层只统计一次
    return d


def calc_array_statistic(tensor, suffix):
    np_array = tensor.numpy()
    if suffix == "avg":
        return np.mean(np_array)
    elif suffix == "var":
        return np.var(np_array)
    elif suffix == "mid":
        return np.median(np_array)
    elif suffix == "max":
        return np.max(np_array)
    elif suffix == "min":
        return np.min(np_array)
    elif suffix == "upper":
        return np.percentile(np_array, 75)
    elif suffix == "lower":
        return np.percentile(np_array, 25)
    elif suffix == "skew":
        return stats.skew(np_array)
    elif suffix == "kurt":
        return stats.kurtosis(np_array)
    elif suffix == "rate0":
        return np.sum(np_array == 0) / np_array.size
    else:
        raise ValueError("metric_suffix should be in ['avg', 'var', 'mid', 'max', 'min', "
                         "'upper', 'lower', 'skew', 'kurt', 'rate0']")


class TrainingHookType(Enum):
    TrainingBeginHook = 0
    TrainingEndHook = 1
    EpochBeginHook = 2
    EpochEndHook = 3
    IterationBeginHook = 4
    IterationEndHook = 5


def sleep(i):
    os.system("sleep " + str(i))
