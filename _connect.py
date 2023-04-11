import sqlite3

import numpy as np
import yaml


def conn():
    # trial_id = "63koyarg"
    # exp_dir = os.path.join("~/nni-experiments/", trial_id)
    # db_dir = os.path.join(exp_dir, "db")
    # db_path_old = os.path.join(db_dir, "nni.sqlite")
    # db_path = os.path.join(db_dir, "nni.db")
    # os.system(" ".join(["cp", db_path_old, db_path]))
    db_path = "/Users/admin/Desktop/nni.sqlite"
    print(db_path)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    # sql = "SELECT tbl_name \
    #         FROM sqlite_master WHERE type = 'table' "
    # cur.execute(sql)
    # values = cur.fetchall()
    # print(values) #[('TrialJobEvent',), ('MetricData',), ('ExperimentProfile',)]

    # sql = "SELECT * \
    #         FROM ExperimentProfile"
    # cur.execute(sql)
    # values = cur.fetchall()
    # d = dict(yaml.load(values[19][0],Loader=yaml.FullLoader))
    # print(d)

    sql = "SELECT * FROM MetricData"  # final result???
    cur.execute(sql)
    values = cur.fetchall()
    d = yaml.load(eval(values[7][5]), Loader=yaml.FullLoader)
    print(d["default"])  # metric
    # print(values[3][2])  # parameterId 'text'
    # print(values[3][4])  # sequence 'integer'
    # print(type(d["module_metric_2da"]), d["module_metric_2da"])
    # print(np.array(d["module_metric_2da"]["__ndarray__"]))
    print(d)

    # param_id_max_step_dict = {}
    # param_id_metric_list_dict = {}
    # max_len = 0
    # for i in range(len(values)):  # final ??? seq include 0
    #     param_id = values[i][2]
    #     metric = yaml.load(eval(values[i][5]), Loader=yaml.FullLoader)["default"]
    #     param_id_max_step_dict[param_id] = values[i][4] if param_id not in param_id_max_step_dict else \
    #         max(param_id_max_step_dict[param_id], values[i][4])
    #     if param_id not in param_id_metric_list_dict:
    #         param_id_metric_list_dict[param_id] = [metric]
    #     else:
    #         param_id_metric_list_dict[param_id].append(metric)
    #         max_len = max(max_len, len(param_id_metric_list_dict[param_id]))
    # for key in param_id_metric_list_dict.keys():
    #     param_id_metric_list_dict[key].pop(-1)
    # max_len -= 1
    # print(param_id_max_step_dict)
    # print(param_id_metric_list_dict)
    # print(max_len)
    #
    # plt.figure(figsize=(20, 12))
    # x = [i for i in range(max_len + 1)]
    # for key in param_id_max_step_dict.keys():
    #     y1 = param_id_metric_list_dict[key]
    #     x1 = x[:param_id_max_step_dict[key] + 1]
    #
    #     y2 = param_id_metric_list_dict[key][:param_id_max_step_dict[key]]
    #     x2 = x[:param_id_max_step_dict[key]]
    #
    #     plt.plot(x1, y1, color='b', marker='o')
    #     plt.plot(x2, y2, color='r', marker='o')
    # plt.show()

    # sql = "PRAGMA table_info(MetricData)"
    # cur.execute(sql)
    # values = cur.fetchall()
    # print(values)
    # [(0, 'timestamp', 'integer', 0, None, 0), (1, 'trialJobId', 'text', 0, None, 0),
    # (2, 'parameterId', 'text', 0, None, 0), (3, 'type', 'text', 0, None, 0),
    # (4, 'sequence', 'integer', 0, None, 0), (5, 'data', 'text', 0, None, 0)]

    # sql = "SELECT * FROM TrialJobEvent"
    # sql = "SELECT data FROM TrialJobEvent WHERE event='WAITING'"
    # cur.execute(sql)
    # values = cur.fetchall()
    # print(values[120][0]) # {"parameter_id": 3, "parameter_source": "algorithm", "parameters": {"bn_layer": false, "act_func": "relu", "grad_clip": false, "init": "xavier", "opt": "adam", "batch_size": 178, "lr": 10443.633034810522, "gamma": 0.46387276784201514, "weight_decay": 0.2528998015083218, "hidden_size": 60, "num_layers": 2}, "parameter_index": 0}
    # d = dict(yaml.load(values[-57][0], Loader=yaml.FullLoader))  # waiting [parameters] / succeeded
    # # print(values[-57])  # 57 waiting
    # print(d["parameters"])

    # sql = "PRAGMA table_info(TrialJobEvent)"
    # cur.execute(sql)
    # values = cur.fetchall()
    # print(values)
    # [(0, 'timestamp', 'integer', 0, None, 0), (1, 'trialJobId', 'text', 0, None, 0), (2, 'event', 'text', 0, None, 0),
    # (3, 'data', 'text', 0, None, 0), (4, 'logPath', 'text', 0, None, 0),
    # (5, 'sequenceId', 'integer', 0, None, 0), (6, 'message', 'text', 0, None, 0)]


if __name__ == '__main__':
    conn()
