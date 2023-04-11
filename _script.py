import os
import time


def main():
    print()
    num = 1
    for i in range(num):  # [0,1,2...9]
        tmp = "./script_log/log_" + time.strftime("%Y-%m-%d--%H-%M-%S", time.localtime())
        os.system("/Library/Frameworks/Python.framework/Versions/3.8/bin/nnictl stop --port 8080")  # 旧进程未完全清理? --port 8080

        # 预防旧进程未完全清理
        os.system("ps -ef | grep python3.8 | \
                        grep peizhon+ | grep -v grep | grep -v script | grep -v view | \
                        awk '{print $2}' | xargs kill -9")

        exp_file_name = "atdd_model_fashionlenet5_monitor_inspector_assessor.yaml"

        exp_name = "_" + exp_file_name.split(".")[0]
        log_file_name = "\"" + tmp + exp_name + "_idx_" + str(i) + ".txt" + "\""

        os.system(
            "/Library/Frameworks/Python.framework/Versions/3.8/bin/nnictl create --config " + exp_file_name + " --port 8080 > " + log_file_name)
        os.system("sleep 1h")
        os.system("sleep 5m")
    # view_lastn_list([1, 2, 3, 4, 5, 6, 7])


if __name__ == '__main__':
    main()
    # plot()
