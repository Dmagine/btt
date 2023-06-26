import sys

sys.path.append("../")
from _common import script_run


def main():
    print()
    # cifar10lstm 105 raw_random raw_gp
    # cifar10lstm 106 our_gp our_smac
    # exchange96auto 101 our_gp our_tpe
    lst = [
        "./cifar10lstm/atdd_model_cifar10lstm_raw_gp.yaml",
        "./cifar10lstm/atdd_model_cifar10lstm_raw_random.yaml",
        "./cifar10lstm/atdd_model_cifar10lstm_raw_gp.yaml",
        "./cifar10lstm/atdd_model_cifar10lstm_raw_random.yaml",
    ]
    t = 6  ####
    script_run(lst, t, log_dir="../script_log/")


if __name__ == '__main__':
    main()
