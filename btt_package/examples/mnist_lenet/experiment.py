import threading
from copy import deepcopy

import torch
from btt.exp_manager import ExperimentConfig, TunerHpoConfig, MonitorRuleConfig
from btt.utils import RecordMode


def get_tuner_config(exp_config):
    exp_config.tuner_config = [TunerHpoConfig() for _ in range(3)]
    exp_config.tuner_config[0].class_name = 'RandomHpo'
    exp_config.tuner_config[1].class_name = 'BatchHpo'
    exp_config.tuner_config[1].exp_dur_not_trial_num = False
    batch_trial_num = 10
    exp_config.tuner_config[1].end_ratio = batch_trial_num / exp_config.max_trial_number  # batch tuner?
    exp_config.tuner_config[2].class_name = 'GpHpo'
    exp_config.tuner_config[2].start_ratio = 0.5

    batch_conf = exp_config.tuner_config[1]
    v1 = {"conv1_k_num": 6, "pool1_size": 3, "conv2_k_num": 17, "pool2_size": 2, "full_num": 84,
          "conv_k_size": 4, "lr": 0.1, "weight_decay": 0.01, "drop_rate": 0.5, "batch_norm": 1, "drop": 1,
          "batch_size": 300, "data_norm": 1, "gamma": 0.7, "step_size": 1, "grad_clip": 1, "clip_thresh": 10,
          "act": 0, "opt": 0, "pool": 0}
    v2 = {"conv1_k_num": 100, "pool1_size": 5, "conv2_k_num": 200, "pool2_size": 3, "full_num": 1000,
          "conv_k_size": 5, "lr": 0.01, "weight_decay": 0.001, "drop_rate": 0.3, "batch_norm": 0, "drop": 0,
          "batch_size": 100, "data_norm": 0, "gamma": 0.9, "step_size": 2, "grad_clip": 0, "clip_thresh": 5,
          "act": 1, "opt": 1, "pool": 1}
    v_list = [deepcopy(v1) for _ in range(10)]
    for k in v2.keys():
        idx = list(v2.keys()).index(k)
        if list(v2.keys()).index(k) == batch_trial_num:
            break
        v_list[idx][k] = v2[k]
    batch_conf.init_args = {'batch_params': v_list}
    return exp_config


def get_monitor_config(exp_config):
    exp_config.monitor_config = []
    rule_name_list = ['train_acc', 'train_loss', 'val_acc', 'val_loss', 'test_acc', 'test_loss',
                      'weight_val', 'weight_grad',
                      'feature_val_in', 'feature_grad_out',
                      'weight_val_abs', 'weight_grad_abs',
                      'feature_val_in_abs', 'feature_grad_out_abs',
                      ]
    for rule_name in rule_name_list:
        conf = MonitorRuleConfig()
        conf.name = rule_name
        conf.intermediate_report = True
        conf.final_report = False
        if ("acc" in rule_name or "loss" in rule_name) \
                and ("train" in rule_name or "val" in rule_name or "test" in rule_name):
            conf.class_name = 'ModePeriodMonitorRule'
            conf.init_args = {'key': rule_name, 'mode_name': RecordMode.EpochTrainEnd.name}
            if "val" in rule_name:
                conf.init_args['mode_name'] = RecordMode.EpochValEnd.name
            elif "test" in rule_name:
                conf.class_name = 'ModeOnceMonitorRule'
                conf.init_args['mode_name'] = RecordMode.EpochEnd.name
                conf.intermediate_report = False
                conf.final_report = True
        elif "weight" in rule_name:
            conf.class_name = 'WeightStatisticsMonitorRule'
            conf.init_args = {'metric_prefix': rule_name, 'calc_batch_ratio': 0.01}
        elif "feature" in rule_name:
            conf.class_name = 'FeatureStatisticsMonitorRule'
            conf.init_args = {'metric_prefix': rule_name, 'calc_batch_ratio': 0.01}
        if rule_name == "val_acc":
            conf.intermediate_default = True
        elif rule_name == "test_acc":
            conf.final_default = True
        exp_config.monitor_config.append(conf)

    return exp_config


def get_assessor_config(exp_config):
    # 先复现吧 规则优化的事情以后再说 （新想法：输入变多，可以比较，细粒度规则）
    exp_config.monitor_config = []
    indicator_name_list = ['AbnormalGradientValues(AGV)', "ExponentiallyAmplifiedGradients(EAG)",
                           "ExponentiallyReducedGradients(ERG)", "PassiveLossChanges(PLC)", "LowActivationRatio(LAR)",
                           "UnexpectedLossChanges(ULC)", "NoMoreGain(NMG)"]
    indicator_class_name_list = ['AgvIndicator', 'EagIndicator', 'ErgIndicator', 'PlcIndicator', 'LarIndicator',
                                 'UlcIndicator', 'NmgIndicator']
    for indicator_name, indicator_class_name in zip(indicator_name_list, indicator_class_name_list):
        conf = MonitorRuleConfig()
        conf.name = indicator_name
        conf.class_name = indicator_class_name
        exp_config.monitor_config.append(conf)

    return exp_config


def main():
    exp_config = ExperimentConfig()
    exp_config.exp_name = 'class_mnist_lenet_test'
    exp_config.exp_description = exp_config.exp_name
    exp_config.trial_concurrency = 2
    exp_config.trial_gpu_number = 1 if torch.cuda.is_available() else 0
    exp_config.max_trial_number = 1000
    exp_config.max_exp_duration = '6h'

    exp_config = get_tuner_config(exp_config)
    exp_config = get_monitor_config(exp_config)
    exp_config = get_assessor_config(exp_config)

    experiment = Experiment(exp_config)
    experiment.start()


if __name__ == '__main__':
    main()
