import os

from _common import tslib_script_run


def main():
    repeat = 3  ####
    hour = 6  ####
    print()
    lst = ["./long_term_forcast/ETTh1/TimesNet/smac_msr.yaml"] * repeat
    lst += ["./long_term_forcast/Traffic/TimesNet/smac_msr.yaml"] * repeat
    lst += ["./long_term_forcast/ETTh1/Transformer/smac_msr.yaml"] * repeat
    lst += ["./long_term_forcast/Traffic/Transformer/smac_msr.yaml"] * repeat
    lst += ["./long_term_forcast/ETTh1/TimesNet/smac_lce.yaml"] * repeat
    lst += ["./long_term_forcast/Traffic/TimesNet/smac_lce.yaml"] * repeat
    lst += ["./long_term_forcast/ETTh1/Transformer/smac_lce.yaml"] * repeat
    lst += ["./long_term_forcast/Traffic/Transformer/smac_lce.yaml"] * repeat
    tslib_script_run(lst, hour, log_dir="_script_log/")


if __name__ == '__main__':
    main()
