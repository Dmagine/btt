import os
import time


def main():
    print()

    d = {
        "atdd_model_mnistlenet_raw_bohb.yaml": 3,
        "atdd_model_mnistlenet_raw_random.yaml": 3,
        "atdd_model_mnistlenet_raw_smac.yaml": 3,
        "atdd_model_mnistlenet_raw_smac_lce.yaml": 3,
        "atdd_model_mnistlenet_raw_tpe.yaml": 3,
        "atdd_model_mnistlenet_raw_tpe_msr.yaml": 3,
        "atdd_model_mnistlenet_atdd_random_inspector_assessor.yaml": 3,
        "atdd_model_mnistlenet_atdd_random_inspector.yaml": 3,
        "atdd_model_mnistlenet_atdd_random_assessor.yaml": 3,
    }

    for k, v in d.items():  # [0,1,2...9]
        exp_file_name = k
        for i in range(v):
            tmp = "../../script_log/log_" + time.strftime("%Y-%m-%d--%H-%M-%S", time.localtime())
            os.system("nnictl stop --port 8080")  # 旧进程未完全清理? --port 8080

            # 预防旧进程未完全清理
            os.system("ps -ef | grep python3.8 | \
                            grep peizhon+ | grep -v grep | grep -v script | grep -v view | \
                            awk '{print $2}' | xargs kill -9")
            exp_name = "_" + exp_file_name.split(".")[0]
            log_file_name = "\"" + tmp + exp_name + "_idx_" + str(i) + ".txt" + "\""

            os.system("nnictl create --config " + exp_file_name + " --port 8080 > " + log_file_name)
            os.system("sleep 3h")  ###########
            os.system("sleep 5m")


if __name__ == '__main__':
    main()
