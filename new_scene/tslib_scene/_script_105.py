from _common import tslib_script_run


def main():
    repeat = 3  ####
    hour = 6  ####
    print()
    lst = ["./long_term_forcast/Traffic/TimesNet-batchsize/random.yaml"] * 1  #######
    tslib_script_run(lst, hour, log_dir="_script_log/")


if __name__ == '__main__':
    main()
