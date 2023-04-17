import sys

sys.path.append("../")
from _common import script_run


def main():
    print()

    lst = [
        "atdd_model_exchange96auto_raw_gp_msr.yaml",
        "atdd_model_exchange96auto_raw_random_msr.yaml",
        "atdd_model_exchange96auto_raw_smac_msr.yaml",
        "atdd_model_exchange96auto_raw_tpe_msr.yaml",
    ]
    t = 6  # t小时 每个实验的时间
    script_run(lst, t)


if __name__ == '__main__':
    main()
