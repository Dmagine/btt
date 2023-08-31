import os
import sqlite3

import numpy as np
import yaml


def get_sqlite_path_list(model_name, dataset_name, hpo_name):
    # /Users/admin/Desktop/TSlib-exp/$model/$dataset/$hpo
    exp_dir = "/Users/admin/Desktop/TSlib-exp/{}/{}/{}/".format(model_name, dataset_name, hpo_name)
    sqlite_path_list = []
    for trial_id in os.listdir(exp_dir):
        sqlite_path = os.path.join(exp_dir, trial_id, "db", "nni.sqlite")
        if os.path.exists(sqlite_path):
            sqlite_path_list.append(sqlite_path)
    # print("get_sqlite_path_list:", sqlite_path_list)
    return sqlite_path_list


def get_specific_final_metric_list(sqlite_path, metric_name="test_data_mse", ):
    conn = sqlite3.connect(sqlite_path)
    cur = conn.cursor()
    sql = "SELECT * FROM MetricData WHERE type='FINAL' "
    cur.execute(sql)
    values = cur.fetchall()
    id_mse_dict = {}
    for i in range(len(values)):
        trial_id = values[i][1]
        d = yaml.load(eval(values[i][5]), Loader=yaml.FullLoader)
        val = float(d["test_data_mse"])
        id_mse_dict[trial_id] = val
    metric_list = list(id_mse_dict.values())
    return metric_list


def calc_top(model_name, dataset_name, hpo_name, top_n, return_mode):
    sqlite_path_list = get_sqlite_path_list(model_name, dataset_name, hpo_name)
    multi_trial_metric_list = []
    for sqlite_path in sqlite_path_list:
        single_trial_metric_list = get_specific_final_metric_list(sqlite_path)
        multi_trial_metric_list.append(np.mean(sorted(single_trial_metric_list)[:top_n]))
    if return_mode == "raw":
        r = multi_trial_metric_list
    elif return_mode == "best":
        r = np.min(multi_trial_metric_list).round(3)
    elif return_mode == "avg":
        r = np.min(multi_trial_metric_list).round(3)
    else:
        raise ValueError("return_mode should be raw, best or avg")
    print("calc_top:", r)
    return r


def calc_tophitrate(model_name, dataset_name, hpo_name, top_n):
    hpo_name1, hpo_name2 = hpo_name.replace("_BTT", ""), hpo_name
    sqlite_path_list1 = get_sqlite_path_list(model_name, dataset_name, hpo_name1)
    sqlite_path_list2 = get_sqlite_path_list(model_name, dataset_name, hpo_name2)

    path_list_dict = {}
    for sqlite_path in sqlite_path_list1 + sqlite_path_list2:
        path_list_dict[sqlite_path] = get_specific_final_metric_list(sqlite_path)  ## no top_n

    for sqlite_path1 in sqlite_path_list1:
        for sqlite_path2 in sqlite_path_list2:
            single_trial_metric_list1 = path_list_dict[sqlite_path1]
            single_trial_metric_list2 = path_list_dict[sqlite_path2]
            metric_list = (single_trial_metric_list1 + single_trial_metric_list2)[:top_n]
            count = 0
            for metric in metric_list:
                if metric in single_trial_metric_list1:
                    count += 1
            r = count / top_n
            print("calc_tophitrate:", r)
            return r


def calc_tsba(model_name, dataset_name, hpo_name):
    hpo_name1, hpo_name2 = hpo_name.replace("_BTT", ""), hpo_name
    sqlite_path_list1 = get_sqlite_path_list(model_name, dataset_name, hpo_name1)
    sqlite_path_list2 = get_sqlite_path_list(model_name, dataset_name, hpo_name2)
    pass


def calc_trial_num(model_name, dataset_name, hpo_name):
    sqlite_path_list = get_sqlite_path_list(model_name, dataset_name, hpo_name)
    trial_num_list = []
    for sqlite_path in sqlite_path_list:
        lst = get_specific_final_metric_list(sqlite_path)
        trial_num_list.append(len(lst))
    r = int(np.mean(trial_num_list))
    print("calc_trial_num:", r)


if __name__ == '__main__':
    # model_name_list = ["TimesNet", "Transformer"]
    # dataset_name_list = ["ETTh1", "Traffic"]
    # hpo_name_list = ["Random", "SMAC", "Random_BTT", "SMAC_BTT"]

    model_name_list = ["TimesNet"]
    dataset_name_list = ["ETTh1"]
    hpo_name_list = ["Random"]

    for model_name in model_name_list:
        for dataset_name in dataset_name_list:
            for hpo_name in hpo_name_list:
                print(model_name, dataset_name, hpo_name)
                calc_top(model_name, dataset_name, hpo_name, 1, "best")
                calc_top(model_name, dataset_name, hpo_name, 1, "avg")
                calc_top(model_name, dataset_name, hpo_name, 10, "avg")
                calc_trial_num(model_name, dataset_name, hpo_name)
                if "_BTT" in hpo_name:
                    calc_tophitrate(model_name, dataset_name, hpo_name, 10)
                    calc_tsba()
                print()
