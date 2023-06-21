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
