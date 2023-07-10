#         self.advisor_config = ATDDMessenger().read_advisor_config()
#         self.shared_config = self.advisor_config["shared"] \
#             if self.advisor_config is not None and "shared" in self.advisor_config else None
#         self.monitor_config = self.advisor_config["monitor"]["classArgs"] \
#             if self.advisor_config is not None and "monitor" in self.advisor_config else None
#         self.assessor_config = self.advisor_config["assessor"]["classArgs"] \
#             if self.advisor_config is not None and "assessor" in self.advisor_config else None
#         self.monitor = BttMonitor(**self.monitor_config) if self.monitor_config is not None else None
import os
import random

import numpy as np
import torch


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


def sleep(i):
    os.system("sleep " + str(i))
