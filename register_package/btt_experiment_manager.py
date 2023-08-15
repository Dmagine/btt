class ExperimentConfig:
    def __init__(self, config_args):
        self.config_args = config_args
        self.exp_name = config_args["exp_name"]
        self.exp_description = config_args["exp_description"]
        self.exp_id = create_exp_uuid()
        self.exp_path = create_exp_path() if "path" not in config_args else config_args["exp_path"]
        self.exp_db = create_exp_db()
        self.trial_concurrency = config_args["trial_concurrency"]
        self.trial_gpu_number = config_args["trial_gpu_number"] if "trial_gpu_number" in config_args else 0
        self.max_trial_number = config_args["max_trial_number"] if "max_trial_number" in config_args else None
        self.max_exp_duration = config_args["max_exp_duration"] if "max_exp_duration" in config_args else None
        self.space_path = config_args["space_path"]

        self.monitor_config = config_args["monitor_config"] if "monitor_config" in config_args else None
        self.tuner_config = config_args["tuner_config"] if "tuner_config" in config_args else None
        self.assessor_config = config_args["assessor_config"] if "assessor_config" in config_args else None
        self._canonicalize()

    def _canonicalize(self):  # "default_config"
        pass

        # experiment = Experiment('local')


#     experiment.config.experiment_name = 'class_mnist_lenet_test'
#     experiment.config.trial_command = 'python3 trial.py'
#     experiment.config.trial_concurrency = 1  ###
#     experiment.config.max_trial_number = 1000
#     experiment.config.max_experiment_duration = '6h'
#
#     f = open('space.json')
#     experiment.config.search_space = json.load(f)
#
#     # experiment.config.advisor. ?
#
#     experiment.config.advisor.name = None
#     experiment.config.advisor.code_directory = '../../register_package'
#     experiment.config.advisor.class_name = 'btt_advisor.BttAdvisor'
#     experiment.config.advisor.class_args = {
#         'monitor': {"classArgs": {"rule_config": {}}},
#         'tuner': {'name': 'random'},
#     }


class BttExperimentManager:
    def __init__(self, exp_config):
        # start stop resume view
        self.exp_name = experiment_name
        self.exp_description = experiment_description
        self.exp_id = create_exp_uuid()
        self.exp_path = experiment_path
