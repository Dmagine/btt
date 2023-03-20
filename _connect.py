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

    sql = "SELECT * FROM MetricData"
    cur.execute(sql)
    values = cur.fetchall()
    d = yaml.load(eval(values[0][5]), Loader=yaml.FullLoader)
    print(d["default"])

    sql = "PRAGMA table_info(MetricData)"
    cur.execute(sql)
    values = cur.fetchall()
    print(values)

    # sql = "SELECT * \
    #         FROM TrialJobEvent"
    # cur.execute(sql)
    # values = cur.fetchall()
    # d = dict(yaml.load(values[0][3], Loader=yaml.FullLoader))  # waiting [parameters] / succeeded
    # # print(d["parameters"])
    # print(values[-70])
    #
    # sql = "PRAGMA table_info(TrialJobEvent)"
    # cur.execute(sql)
    # values = cur.fetchall()
    # print(values)


def calc_top():



    pass


if __name__ == '__main__':
    conn()
