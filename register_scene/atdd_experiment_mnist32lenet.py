import nni
import torch
from nni.experiment import Experiment


def main():
    experiment = Experiment('local')
    experiment.config.experiment_name = 'mnist32lenet_test'
    experiment.config.trial_command = 'python3 atdd_model_mnist32lenet.py'
    experiment.config.trial_concurrency = 2
    experiment.config.max_trial_number = 10
    experiment.config.max_experiment_duration = '1h'

    experiment.config.search_space = {
        "conv1_k_num": {
            "_type": "randint",
            "_value": [
                5,
                500
            ]
        },
        "pool1_size": {
            "_type": "randint",
            "_value": [
                2,
                5
            ]
        },
        "conv2_k_num": {
            "_type": "randint",
            "_value": [
                5,
                500
            ]
        },
        "conv_k_size": {
            "_type": "randint",
            "_value": [
                2,
                6
            ]
        },
        "pool2_size": {
            "_type": "randint",
            "_value": [
                2,
                5
            ]
        },
        "full_num": {
            "_type": "randint",
            "_value": [
                1,
                4096
            ]
        },
        "lr": {
            "_type": "loguniform",
            "_value": [
                0.000001,
                10
            ]
        },
        "weight_decay": {
            "_type": "loguniform",
            "_value": [
                0.000001,
                0.1
            ]
        },
        "batch_norm": {
            "_type": "choice",
            "_value": [
                0,
                1
            ]
        },
        "drop": {
            "_type": "choice",
            "_value": [
                0,
                1
            ]
        },
        "batch_size": {
            "_type": "randint",
            "_value": [
                2,
                2000
            ]
        },
        "data_norm": {
            "_type": "choice",
            "_value": [
                0,
                1
            ]
        },
        "gamma": {
            "_type": "uniform",
            "_value": [
                0,
                1
            ]
        },
        "step_size": {
            "_type": "randint",
            "_value": [
                1,
                7
            ]
        },
        "grad_clip": {
            "_type": "choice",
            "_value": [
                0,
                1
            ]
        },
        "clip_thresh": {
            "_type": "loguniform",
            "_value": [
                0.1,
                100
            ]
        },
        "act": {
            "_type": "choice",
            "_value": [
                0,
                1,
                2,
                3,
                4
            ]
        },
        "opt": {
            "_type": "choice",
            "_value": [
                0,
                1,
                2,
                3,
                4
            ]
        },
        "pool": {
            "_type": "choice",
            "_value": [
                0,
                1
            ]
        }
    }

    # experiment.config.tuner.name = 'random'
    # experiment.config.tuner.class_args = {
    #     'optimize_mode': 'maximize'
    # }

    experiment.config.advisor.code_directory = '../register_package'
    experiment.config.advisor.class_name = 'atdd_advisor.ATDDAdvisor'
    experiment.config.advisor.class_args = {
        'shared': {
            'max_epoch': 20
        },
        'monitor': 'default',
        'tuner': {
            'name': 'random'
        },
        'inspector': 'default',
        'assessor': 'default'
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
    print(f"Using {device} device")
    if device == "cuda":
        experiment.config.trial_gpu_number = 1
        experiment.config.training_service.use_active_gpu = True

    experiment.run(8080, wait_completion=False)
    exp_id = nni.get_experiment_id()
    print(f"Experiment id: {exp_id}")
    # experiment.stop()
    # experiment.view(exp_id,8080)


if __name__ == '__main__':
    main()
