
import sys

sys.path.append("../")
from _common import script_run


def main():
    print()

    lst = [
        "atdd_model_cifar10lstm_raw_gp_lce.yaml",
        "atdd_model_cifar10lstm_raw_random_lce.yaml",
        "atdd_model_cifar10lstm_raw_smac_lce.yaml",
        "atdd_model_cifar10lstm_raw_tpe_lce.yaml",
    ]
    t = 6  # t小时 每个实验的时间
    script_run(lst, t)


if __name__ == '__main__':
    main()