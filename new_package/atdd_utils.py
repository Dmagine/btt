import random
from collections import namedtuple

import numpy as np
import torch

diagnose_params = namedtuple("diagnose_params", {
    "p_vg1", "p_vg2", "p_vg3",
    "p_eg1", "p_eg2", "p_eg3",
    "p_dr1", "p_dr2",
    "p_sc1", "p_sc2",
    "p_ho1", "p_ho2",
    "p_ng1", "p_ng2",
    "wd"
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