import random
from collections import namedtuple

import numpy as np
import torch

diagnose_params = namedtuple("diagnose_params", {
    "alpha1", "alpha2", "alpha3", "beta1", "beta2", "beta3",
    "gamma", "delta", "zeta", "theta", "eta",
    "dd_max_threshold", "dd_min_threshold", "dd_threshold_VG",
    "window_size_float"
})


def get_ave(lst):
    return sum(lst) / len(lst)


def set_seed(seed, msg="",logger=None):
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
