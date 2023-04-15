import os
import time


def script_run(lst, t):
    for i in range(len(lst)):  # [0,1,2...9]
        exp_file_name = lst[i]
        tmp = "../../script_log/log_" + time.strftime("%Y-%m-%d--%H-%M-%S", time.localtime())
        os.system("nnictl stop --port 8080")  # ~/.local/bin/

        # 预防旧进程未完全清理
        os.system("ps -ef | grep python | \
                                    grep peizhon+ | grep -v grep | grep -v script | grep -v view | \
                                    awk '{print $2}' | xargs kill -9")
        os.system("ps -ef | grep python | \
                                    grep cenzhiy+ | grep -v grep | grep -v script | grep -v view | \
                                    awk '{print $2}' | xargs kill -9")
        exp_name = "_" + exp_file_name.split(".")[0]
        log_file_name = "\"" + tmp + exp_name + "_idx_" + str(i) + ".txt" + "\""

        os.system("nnictl create --config " + exp_file_name + " --port 8080 > " + log_file_name)
        os.system("sleep " + str(t) + "h")  ###########
        os.system("sleep 5m")
