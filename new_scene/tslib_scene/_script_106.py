from _common import tslib_script_run


def main():
    repeat = 3  ####
    hour = 6  ####
    print()
    lst = ["./long_term_forcast/Traffic/Transformer/smac.yaml"] * repeat
    lst += ["./long_term_forcast/ETTh1/Transformer/smac.yaml"] * repeat
    tslib_script_run(lst, hour, log_dir="_script_log/")


if __name__ == '__main__':
    main()
