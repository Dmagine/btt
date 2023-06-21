import os
import random
from collections import namedtuple
from enum import Enum

import numpy as np
import torch
import yaml

diagnose_params = namedtuple("diagnose_params", {
    "p_eg1", "p_eg2", "p_eg3",
    "p_vg1", "p_vg2", "p_vg3", "p_vg4",
    "p_dr1", "p_dr2", "p_dr3",
    "p_sc1", "p_sc2", "p_sc3",
    "p_ho1", "p_ho2",
    "p_nmg1", "p_nmg2",
    "wd_ho", "wd_nmg",
})


def get_ave(lst):
    return sum(lst) / len(lst)


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

class RuleType(Enum):
    TRIAL = 1
    EXPERIMENT = 2
    JOINT = 3


def register_rule(name, rule_type, rule_path, class_name):
    yaml_path = "registered_rules.yaml"
    new_rule = {
        'ruleName': name,
        'ruleType': rule_type,
        'filePath': rule_path,
        'className': class_name
    }
    # 验证在filePath下是否存在className的类
    if not os.path.exists(rule_path):
        print("rule file not exists!", rule_path)
        return
    if not os.path.exists(yaml_path):
        with open(yaml_path, "w") as f:
            rules = {[new_rule]}
            yaml.dump(rules, f)
    else:
        with open(yaml_path, "r") as f:
            rules = yaml.load(f.read(), Loader=yaml.FullLoader)
            for rule in rules:
                if rule['ruleName'] == name:
                    print("rule already exists!", name)
                    return
            rules.append(new_rule)
        with open(yaml_path, "w") as f:
            yaml.dump(rules, f)
    print("rule registered:", name)


def unregister_rule(name):
    yaml_path = "registered_rules.yaml"
    if not os.path.exists(yaml_path):
        print("no rule registered!")
        return
    with open(yaml_path, "r") as f:
        rules = yaml.load(f.read(), Loader=yaml.FullLoader)
        for rule in rules:
            if rule['ruleName'] == name:
                rules.remove(rule)
                break
    with open(yaml_path, "w") as f:
        yaml.dump(rules, f)
        print("rule unregistered:", name)


if __name__ == "__main__":
    register_rule("test", RuleType.TRIAL, "test", "test")
    register_rule("test2", RuleType.TRIAL, "test2", "test2")
    register_rule("test3", RuleType.TRIAL, "test3", "test3")
    register_rule("test3", RuleType.TRIAL, "test3", "test3")
    unregister_rule("test")
