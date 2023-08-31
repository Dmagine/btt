from btt.exp_manager import BttExperimentManager

from trial import trial_func


def hpo_exp_main():
    exp_manager = BttExperimentManager()
    exp_manager.trial_func = trial_func
    exp_manager.start()


if __name__ == '__main__':
    hpo_exp_main()
