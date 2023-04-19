import sys

sys.path.append("../")
from _common import script_run


def main():
    print()

    lst = [
        "./exchange96auto/atdd_model_exchange96auto_monitor.yaml",
        "./cifar10cnn/atdd_model_cifar10cnn_monitor.yaml",
    ]
    t = 12  ####
    script_run(lst, t, log_dir="../script_log/")


if __name__ == '__main__':
    main()
