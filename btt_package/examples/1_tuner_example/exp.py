from btt.exp_manager import BttExperimentManager, ExperimentConfig, \
    TunerHpoConfig, Hyperparameter, ParamMode, MonitorRuleConfig
from btt.monitor.monitor_rule import ModePeriodRule, ModeOnceRule
from btt.tuner.tuner_hpo import RandomHpo
from btt.utils import RecordMode

from trial import trial_func


def hpo_exp_main():
    exp_config = ExperimentConfig()
    exp_config.tuner_config = {"my_random": TunerHpoConfig(RandomHpo)}
    exp_config.hp_config = {
        "lr": Hyperparameter(ParamMode.Choice, [0.01, 0.001, 0.0001], 0.001, "learning rate"),
        "batch_size": Hyperparameter(ParamMode.Choice, [32, 64, 128], 64, "training and test data batch size"),
    }
    exp_config.monitor_config = {
        # rule_name 在canonicalize中会注入进 instance
        # default 在canonicalize中会带 report
        "val_loss": MonitorRuleConfig(ModePeriodRule, RecordMode.EpochTrainEnd, interm_report=True),
        "val_acc": MonitorRuleConfig(ModePeriodRule, RecordMode.EpochTrainEnd, interm_default=True),
        "test_loss": MonitorRuleConfig(ModeOnceRule, RecordMode.EpochTrainEnd, final_report=True),
        "test_acc": MonitorRuleConfig(ModeOnceRule, RecordMode.EpochTrainEnd, final_default=True),
    }

    exp_manager = BttExperimentManager(exp_config)
    exp_manager.trial_func = trial_func
    exp_manager.start()


if __name__ == '__main__':
    hpo_exp_main()
