import os
import time


def main():
    repeat = 5 - 2  ####
    hour = 6  ####
    print()
    lst = ["../tslib_scene/long_term_forcast/ETTh1/TimesNet/raw_random.yaml"] * repeat  #######
    # 30s/epoch 20epoch 10min/trial 4parallel 2.5m/trial 100trial->250m->4h->6h
    # 最新：20-90min/trial 1h/trial 4parallel 100trial->25h->20h 扎实但难调整。。。
    # 不挑了 （6h5次！）（说不定时间约少 工具效果越明显呢）
    tslib_script_run(lst, hour, log_dir="./_script_log/")


def tslib_script_run(lst, hour, log_dir=None):
    for i in range(len(lst)):  # [0,1,2...9]
        exp_file_path = lst[i]
        log_dir = "./_script_log/" if log_dir is None else log_dir  ####
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        tmp = log_dir + ("./log_" + time.strftime("%Y-%m-%d--%H-%M-%S", time.localtime()))
        os.system("nnictl stop --port 8080")  # ~/.local/bin/

        res = os.system("ps -ef | grep cenzhiy+ | " +
                        "grep python3 | grep trial.py | " +  # 条件 trial_command
                        "grep " + str(__file__.split("/")[-1]) + " | " +  # 反过滤 _script_tslib.py
                        "grep -v grep | grep -v _script | grep -v view | " +  # trial_command
                        "awk '{print $2}' | xargs kill -9")
        if i != 0:
            os.system("sleep 10m")  # gpu休息
        exp_name = "_" + exp_file_path.split("/")[-1].split(".")[0]
        log_note = "_".join(exp_file_path.split(".")[-2].split("/")[-4:])
        log_file_name = "\"" + tmp + "_" + log_note + "_idx_" + str(i) + ".txt" + "\""
        cmd = "nnictl create --config " + exp_file_path + " --port 8080 > " + log_file_name
        res = os.system(cmd)
        os.system("sleep " + str(hour) + "h")


if __name__ == '__main__':
    main()
