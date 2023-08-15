import json
import os
import sys
import threading
from copy import deepcopy

import torch
from nni.experiment import Experiment

sys.path.append("../../register_package")
from btt_utils import RecordMode


def main():
    experiment = Experiment('local')
    experiment.config.experiment_name = 'class_mnist_lenet_test'
    experiment.config.trial_command = 'python3 trial.py'
    experiment.config.trial_concurrency = 1  ###
    experiment.config.max_trial_number = 1000
    experiment.config.max_experiment_duration = '6h'

    f = open('space.json')
    experiment.config.search_space = json.load(f)

    # experiment.config.advisor. ?

    experiment.config.advisor.name = None
    experiment.config.advisor.code_directory = '../../register_package'
    experiment.config.advisor.class_name = 'btt_advisor.BttAdvisor'

    experiment.config.advisor.class_args = {
        'monitor': {"classArgs": {"rule_config": {}}},
        'tuner': {'name': 'random'},
    }
    rule_name_list = ['train_acc', 'train_loss', 'val_acc', 'val_loss', 'test_acc', 'test_loss',
                      'weight_val', 'weight_grad',
                      'feature_val_in', 'feature_grad_out']

    for rule_name in rule_name_list:
        d = {}
        d.update({
            "mode": "rule",
            "code_dir": "../../register_package/",
            "module_name": "btt_monitor_rule",
            "intermediate_report": True,
            "final_report": False,
        })
        if ("acc" in rule_name or "loss" in rule_name) \
                and ("train" in rule_name or "val" in rule_name or "test" in rule_name):
            d.update({"class_name": "ModePeriodMonitorRule",
                      "init_args": {"key": rule_name, "mode_name": RecordMode.EpochTrainEnd.name}})
            if "val" in rule_name:
                d['init_args']['mode_name'] = RecordMode.EpochValEnd.name
            elif "test" in rule_name:
                d.update({"class_name": "ModeOnceMonitorRule",
                          "init_args": {"key": rule_name, "mode_name": RecordMode.End.name},
                          "intermediate_report": False, "final_report": True})
        elif "weight" in rule_name:
            d.update({"class_name": "WeightStatisticsMonitorRule",
                      "init_args": {"metric_prefix": rule_name, "calc_batch_ratio": 0.01}})
        elif "feature" in rule_name:
            d.update({"class_name": "FeatureStatisticsMonitorRule",
                      "init_args": {"metric_prefix": rule_name, "calc_batch_ratio": 0.01}})
        if rule_name == "val_acc":
            d['intermediate_default'] = True
        elif rule_name == "test_acc":
            d['final_default'] = True
        experiment.config.advisor.class_args['monitor']['classArgs']['rule_config'][rule_name] = d

    batch_tuner = True
    if batch_tuner:
        experiment.config.advisor.class_args['tuner']['name'] = 'Batch'
        v1 = {"conv1_k_num": 6, "pool1_size": 3, "conv2_k_num": 17, "pool2_size": 2, "full_num": 84, "conv_k_size": 4,
              "lr": 0.1, "weight_decay": 0.01, "drop_rate": 0.5, "batch_norm": 1, "drop": 1, "batch_size": 300,
              "data_norm": 1, "gamma": 0.7, "step_size": 1, "grad_clip": 1, "clip_thresh": 10, "act": 0, "opt": 0,
              "pool": 0}
        v_list = [deepcopy(v1) for _ in range(10)]
        v_list[0]['conv1_k_num'] = 100
        v_list[1]['pool1_size'] = 5
        v_list[2]['conv2_k_num'] = 200
        v_list[3]['pool2_size'] = 3
        v_list[4]['full_num'] = 1000
        v_list[5]['conv_k_size'] = 5
        v_list[6]['lr'] = 0.01
        v_list[7]['weight_decay'] = 0.001
        v_list[8]['drop_rate'] = 0.3
        v_list[9]['batch_norm'] = 0
        experiment.config.search_space = {
            'combine_params': {
                '_type': 'choice',
                '_value': v_list
            }
        }

    # experiment.config.trial:
    #   command: python3 atdd_model_mnist32lenet.py
    #   codeDirectory: ../../new_package ###
    #   gpuNum: 1
    #   cpuNum: 1
    #   memoryMB: 4096
    #   gpuType: 1080ti
    #   gpuIndices: [0]
    #   nniManagerPort: 8081
    #   nniManagerIP:

    experiment.config.training_service.platform = 'local'

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        experiment.config.trial_gpu_number = 1
        experiment.config.training_service.use_active_gpu = True

    exp_id = experiment.id
    d = {"device": device, "exp_id": exp_id}
    threading.Thread(target=des, kwargs=d).start()
    experiment.run(8080)  # background
    # experiment.start(8080)

    # experiment.stop()
    # experiment.view(exp_id,8080)


def des(device, exp_id):
    os.system("sleep 5")
    print(f"Using {device} device")
    print(f"Experiment id: {exp_id}")


if __name__ == '__main__':
    main()
