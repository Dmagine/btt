import sqlite3

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

    # sql = "SELECT * FROM MetricData"
    # cur.execute(sql)
    # values = cur.fetchall()
    # d = yaml.load(eval(values[0][5]), Loader=yaml.FullLoader)
    # print(d["default"])
    #
    # sql = "PRAGMA table_info(MetricData)"
    # cur.execute(sql)
    # values = cur.fetchall()
    # print(values)

    # sql = "SELECT * FROM TrialJobEvent"
    sql = "SELECT data FROM TrialJobEvent WHERE event='WAITING'"
    cur.execute(sql)
    values = cur.fetchall()
    print(values[120][0]) # {"parameter_id": 3, "parameter_source": "algorithm", "parameters": {"bn_layer": false, "act_func": "relu", "grad_clip": false, "init": "xavier", "opt": "adam", "batch_size": 178, "lr": 10443.633034810522, "gamma": 0.46387276784201514, "weight_decay": 0.2528998015083218, "hidden_size": 60, "num_layers": 2}, "parameter_index": 0}
    d = dict(yaml.load(values[-57][0], Loader=yaml.FullLoader))  # waiting [parameters] / succeeded
    # print(values[-57])  # 57 waiting
    print(d["parameters"])

    # sql = "PRAGMA table_info(TrialJobEvent)"
    # cur.execute(sql)
    # values = cur.fetchall()
    # print(values)


if __name__ == '__main__':
    conn()
