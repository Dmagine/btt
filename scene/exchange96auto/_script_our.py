import sys

sys.path.append("../")
from _common import script_run


def main():
    print()

    lst = [
        "atdd_model_exchange96auto_our_gp.yaml",
        "atdd_model_exchange96auto_our_random.yaml",
        "atdd_model_exchange96auto_our_smac.yaml",
        "atdd_model_exchange96auto_our_tpe.yaml",
    ]
    t = 6  # t小时 每个实验的时间
    script_run(lst, t)


if __name__ == '__main__':
    main()
