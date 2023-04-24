import os
import time


def script_run(lst, t, log_dir=None):
    for i in range(len(lst)):  # [0,1,2...9]
        exp_file_path = lst[i]
        log_dir = "../../script_log/" if log_dir is None else log_dir
        tmp = log_dir + ("./log_" + time.strftime("%Y-%m-%d--%H-%M-%S", time.localtime()))
        os.system("nnictl stop --port 8080")  # ~/.local/bin/

        # 预防旧进程未完全清理
        res = os.system("ps -ef | grep python3 | grep atdd | \
                                    grep peizhon+ | grep -v grep | grep -v _script | grep -v view | \
                                    awk '{print $2}' | xargs kill -9")
        res = os.system("ps -ef | grep python3 | grep atdd | \
                                    grep cenzhiy+ | grep -v grep | grep -v _script | grep -v view | \
                                    awk '{print $2}' | xargs kill -9")
        exp_name = "_" + exp_file_path.split("/")[-1].split(".")[0]
        log_file_name = "\"" + tmp + exp_name + "_idx_" + str(i) + ".txt" + "\""
        cmd = "nnictl create --config " + exp_file_path + " --port 8080 > " + log_file_name
        res = os.system(cmd)
        os.system("sleep " + str(t) + "h")  ###########
        os.system("sleep 5m")
