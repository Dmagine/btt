import random
from collections import namedtuple

import numpy as np
import torch

diagnose_params = namedtuple("diagnose_params", {
    "alpha1", "alpha2", "alpha3", "beta1", "beta2", "beta3",
    "gamma", "delta", "zeta", "theta", "eta",
    "dd_max_threshold", "dd_min_threshold", "dd_threshold_VG",
    "window_size_float", "window_size_min"
})

# default_json = {
#     'shared':
#         {'max_epoch': 20,
#          'model_num': 1,
#          'enable_dict': {'acc': True,
#                          'loss': True,
#                          'reward': False,
#                          'model': True,
#                          'opt': True,
#                          'epoch': True,
#                          'val': True,
#                          'test': True,
#                          'data': True,
#                          'lr': True}},
#     'monitor': {'classArgs': {'report': {'intermediate_default': 'val_acc', 'final_default': 'val_acc'}}},
#     'tuner': {'classArgs': {
#         'basic': {'optimize_mode': 'maximize', 'algorithm_name': 'random', 'rule_name_list': ['at', 'dd', 'wd']},
#         'rectify': {'k': 1,
#                     'base_probability': 0.1,
#                     'max_probability': 0.95,
#                     'start_duration_float': 0.9,
#                     'end_duration_float': 1.0,
#                     'same_retry_maximum': 10,
#                     'retained_parameter_list': ['bn_layer',
#                                                 'act_func',
#                                                 'grad_clip',
#                                                 'init',
#                                                 'opt',
#                                                 'batch_size',
#                                                 'lr',
#                                                 'epoch']},
#         'parallel': {'parallel_optimize': False, 'constant_liar_type': 'min'}}},
#     'assessor': {'classArgs': {
#         'basic': {'max_epoch': 20,
#                   'maximize_metric_name_list': ['val_acc', 'sc_metric'],
#                   'minimize_metric_name_list': ['val_loss',
#                                                 'veg_metric',
#                                                 'dr_metric',
#                                                 'ol_metric']},
#         'compare': {'k': 1,
#                     'max_percentile': 90,
#                     'start_step_float': 0.0,
#                     'end_step_float': 0.5,
#                     'metric_demarcation_num': 10, #####
#                     'comparable_trial_minimum': 2},
#         'diagnose': {'beta1': 0.001,
#                      'beta3': 70,
#                      'zeta': 0.03,
#                      'window_size_float': 0.25}}},
#     'inspector': {'classArgs': {
#         'basic': {'max_epoch': 20,'start_step_float': 0.15,'rule_name_list': ['at', 'dd', 'wd']},
#         'diagnose': {'alpha1': 0,
#                      'alpha2': 0,
#                      'alpha3': 0,
#                      'beta1': 0.001,
#                      'beta2': 0.0001,
#                      'beta3': 70,
#                      'gamma': 0.7,
#                      'delta': 0.01,
#                      'zeta': 0.03,
#                      'theta': 0.9,
#                      'eta': 0.2,
#                      'dd_max_threshold': 10,
#                      'dd_min_threshold': 1e-05,
#                      'dd_threshold_VG': 1e-07,
#                      'window_size_float': 0.25}}}}


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
