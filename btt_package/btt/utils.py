import hashlib
import importlib
import os
import random
import uuid
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


def calc_array_statistic(array, suffix):
    if suffix == "avg":
        return np.mean(array)
    elif suffix == "var":
        return np.var(array)
    elif suffix == "mid":
        return np.median(array)
    elif suffix == "max":
        return np.max(array)
    elif suffix == "min":
        return np.min(array)
    elif suffix == "upper":
        return np.percentile(array, 75)
    elif suffix == "lower":
        return np.percentile(array, 25)
    elif suffix == "skew":
        return stats.skew(array)
    elif suffix == "kurt":
        return stats.kurtosis(array)
    elif suffix == "rate0":
        return np.sum(array == 0) / array.size
    else:
        raise ValueError("metric_suffix should be in ['avg', 'var', 'mid', 'max', 'min', "
                         "'upper', 'lower', 'skew', 'kurt', 'rate0']")


class RecordMode(Enum):
    Begin = 0
    EpochBegin = 1
    EpochEnd = 2
    EpochTrainBegin = 3
    EpochTrainEnd = 4
    TrainIterBegin = 5
    TrainIterEnd = 6
    EpochValBegin = 7
    EpochValEnd = 8
    ValIterBegin = 9
    ValIterEnd = 10
    End = 11

    def __json__(self):
        return self.name


class ObtainMode(Enum):
    IdxImmediate = 0
    IdxWait = 1
    AllWait = 2

    def __json__(self):
        return self.name


class ParamMode(Enum):
    Choice = 0
    RandInt = 1
    Uniform = 2
    QUniform = 3
    LogUniform = 4
    QLogUniform = 5
    Normal = 6
    QNormal = 7
    LogNormal = 8
    QLogNormal = 9





    def __json__(self):
        return self.name


def get_uuid(str_len=8):
    return hashlib.md5(str(uuid.uuid4()).encode('utf-8')).hexdigest()[:str_len]


def get_package_abs_dir():
    return importlib.import_module("btt").__path__[0]


def time_str2second(time_str):
    # e.g.: 1d, 2h, 3m or 4s
    if time_str[-1] == "d":
        return int(time_str[:-1]) * 24 * 60 * 60
    elif time_str[-1] == "h":
        return int(time_str[:-1]) * 60 * 60
    elif time_str[-1] == "m":
        return int(time_str[:-1]) * 60
    elif time_str[-1] == "s":
        return int(time_str[:-1])
    else:
        raise ValueError("time_str should be end with 'd', 'h', 'm' or 's'")


def if_any_nan_or_inf(x):
    return np.isnan(x).any or np.isinf(x).any


def str2second(str_time):
    if type(str_time) is int:
        return str_time
    if type(str_time) is str:
        if str_time[-1] == "s":
            return int(str_time[:-1])
        elif str_time[-1] == "m":
            return int(str_time[:-1]) * 60
        elif str_time[-1] == "h":
            return int(str_time[:-1]) * 60 * 60
    raise ValueError("str_time should be int or str with suffix s/m/h")
