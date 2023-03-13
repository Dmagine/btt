import os
from datetime import datetime
import time
from _view import view_lastn_list


def main():
    print()
    os.system("sleep 40m")
    num = 9
    for i in range(num):  # [0,1,2...9]
        tmp = "./script_log/log_" + time.strftime("%Y-%m-%d--%H-%M-%S", time.localtime())
        os.system("nnictl stop --port 8080")  # 旧进程未完全清理? --port 8080

        # 预防旧进程未完全清理
        os.system("ps -ef | grep python3.8 | \
                        grep peizhon+ | grep -v grep | grep -v script | grep -v view | \
                        awk '{print $2}' | xargs kill -9")

        exp_file_name = "atdd_yaml.yaml"

        exp_name = "_" + exp_file_name.split(".")[0]
        log_file_name = "\"" + tmp + exp_name + "_idx_" + str(i) + ".txt" + "\""

        os.system("nnictl create --config " + exp_file_name + " --port 8080 > " + log_file_name)
        os.system("sleep 1h")
        os.system("sleep 5m")
    # view_lastn_list([1, 2, 3, 4, 5, 6, 7])


if __name__ == '__main__':
    main()
    # cal()
