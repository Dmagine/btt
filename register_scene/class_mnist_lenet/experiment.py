import json
import os
import threading

import torch
from nni.experiment import Experiment


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
        'monitor': {
            "classArgs": {
                "rule_config": {
                    "train_acc": {
                        "mode": "period",
                        "intermediate_report": True,
                        "final_report": True
                    },
                    "train_loss": {
                        "mode": "period",
                        "intermediate_report": True,
                        "final_report": True
                    },
                    "valid_acc": {
                        "mode": "period",
                        "intermediate_report": True,
                        "final_report": True,
                        "intermediate_default": True  ###
                    },
                    "valid_loss": {
                        "mode": "period",
                        "intermediate_report": True,
                        "final_report": True
                    },
                    "test_acc": {
                        "mode": "once",
                        "intermediate_report": False,
                        "final_report": True,
                        "final_default": True
                    },
                    "test_loss": {
                        "mode": "once",
                        "intermediate_report": False,
                        "final_report": True
                    },
                    "weight_val": {  # 并行进行 loss.backward 和 提取weight值 存在风险 / deepcopy 消耗时间长 state_dict 格式变不同
                        "mode": "rule",
                        "code_dir": "../../register_package/",
                        "module_name": "btt_monitor_rule",
                        "class_name": "NotImplementMonitorRule", # WeightStatisticsMonitorRule
                        # "init_args": {
                        #     "metric_prefix": "weight_val",
                        # },
                        "intermediate_report": True,
                        "final_report": False
                    },
                    "weight_grad": {  # 动态计算获得grad deepcopy后失效！！！！
                        "mode": "rule",
                        "code_dir": "../../register_package/",
                        "module_name": "btt_monitor_rule",
                        "class_name": "NotImplementMonitorRule", # WeightStatisticsMonitorRule
                        # "init_args": {
                        #     "metric_prefix": "weight_grad",
                        # },
                        "intermediate_report": True,
                        "final_report": False
                    },
                    "feature_val_in": {
                        "mode": "rule",
                        "code_dir": "../../register_package/",
                        "module_name": "btt_monitor_rule",
                        "class_name": "NotImplementMonitorRule",
                        # "class_name": "FeatureStatisticsMonitorRule",
                        # "init_args": {
                        #     "metric_prefix": "feature_val_in",
                        # },
                        "intermediate_report": True,
                        "final_report": False
                    },
                    "feature_grad_out": {
                        "mode": "rule",
                        "code_dir": "../../register_package/",
                        "module_name": "btt_monitor_rule",
                        "class_name": "NotImplementMonitorRule",
                        # "class_name": "FeatureStatisticsMonitorRule",
                        # "init_args": {
                        #     "metric_prefix": "feature_grad_out",
                        # },
                        "intermediate_report": True,
                        "final_report": False
                    }
                }
            }
        },
        'tuner': {
            'name': 'random'
        },
        # 'assessor': 'default'
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
