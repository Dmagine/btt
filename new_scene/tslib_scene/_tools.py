import os
import pickle
import sqlite3

import numpy as np
import yaml


def get_sqlite_path_list(model_name, dataset_name, hpo_name):
    # /Users/admin/Desktop/TSlib-exp/$model/$dataset/$hpo
    exp_dir = "/Users/admin/Desktop/TSlib-exp/{}/{}/{}/".format(model_name, dataset_name, hpo_name)
    # print("exp_dir:", exp_dir)
    sqlite_path_list = []
    for trial_id in os.listdir(exp_dir):
        sqlite_path = os.path.join(exp_dir, trial_id, "db", "nni.sqlite")
        if os.path.exists(sqlite_path):
            sqlite_path_list.append(sqlite_path)
    # print("get_sqlite_path_list:", sqlite_path_list)
    return sqlite_path_list


def get_val(d, k):
    if k == "test_loss":
        for k in ["test_data_mse", "test_loss", "default"]:
            if k in d:
                return float(d[k])
        raise ValueError("no test_loss")
    elif k == "val_loss":
        for k in ["val_data_train_loss", "val_loss"]:
            if k in d:
                return float(d[k])
        raise ValueError("no val_loss")
    elif k == "train_loss":
        for k in ["train_data_train_loss", "train_loss", "default"]:
            if k in d:
                return float(d[k])
        raise ValueError("no train_loss")
    else:
        raise ValueError("k should be test_loss, val_loss or train_loss")


def get_periodical_dict(sqlite_path):
    # 1 get values from pickle or 2 load from sqlite and save values to pickle
    pkl_path = sqlite_path.replace("nni.sqlite", "periodical_dict.pkl")
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            d = pickle.load(f)
            return d
    else:
        conn = sqlite3.connect(sqlite_path)
        cur = conn.cursor()
        sql = "SELECT * FROM MetricData WHERE type='PERIODICAL' "
        cur.execute(sql)
        values = cur.fetchall()
        for i in range(len(values)):
            values[i] = list(values[i])
            values[i][5] = yaml.load(eval(values[i][5]), Loader=yaml.FullLoader)
        with open(pkl_path, "wb") as f:
            pickle.dump(values, f)
        return values


def get_final_dict(sqlite_path):
    pkl_path = sqlite_path.replace("nni.sqlite", "final_dict.pkl")
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            d = pickle.load(f)
            return d
    else:
        conn = sqlite3.connect(sqlite_path)
        cur = conn.cursor()
        sql = "SELECT * FROM MetricData WHERE type='FINAL' "
        cur.execute(sql)
        values = list(cur.fetchall())
        for i in range(len(values)):
            values[i] = list(values[i])
            values[i][5] = yaml.load(eval(values[i][5]), Loader=yaml.FullLoader)
        with open(pkl_path, "wb") as f:
            pickle.dump(values, f)
        return values


def get_specific_last_period_metric_dict(sqlite_path, key="test_loss"):
    d = get_periodical_dict(sqlite_path)
    id_epoch_dict = {}  # last epoch idx 19
    rd = {}
    for i in range(len(d)):
        trial_id = d[i][1]
        if trial_id not in id_epoch_dict:
            id_epoch_dict[trial_id] = 0
        id_epoch_dict[trial_id] += 1
        rd[trial_id] = get_val(d[i][5], key)  # 覆盖
    return rd


def get_specific_final_metric_dict(sqlite_path, key="test_loss"):
    d = get_final_dict(sqlite_path)
    rd = {}
    for i in range(len(d)):
        trial_id = d[i][1]
        rd[trial_id] = get_val(d[i][5], key)
    return rd


def calc_top(model_name, dataset_name, hpo_name, top_n, return_mode):
    sqlite_path_list = get_sqlite_path_list(model_name, dataset_name, hpo_name)
    multi_trial_metric_list = []
    for sqlite_path in sqlite_path_list:
        single_trial_metric_list = list(get_specific_final_metric_dict(sqlite_path).values())
        multi_trial_metric_list.append(np.mean(sorted(single_trial_metric_list)[:top_n]))
    if return_mode == "raw":
        r = multi_trial_metric_list
    elif return_mode == "best":
        r = np.min(multi_trial_metric_list).round(3)
    elif return_mode == "avg":
        r = np.mean(multi_trial_metric_list).round(3)
    else:
        raise ValueError("return_mode should be raw, best or avg")
    print("calc_top:", r)
    return r


def calc_tophitrate(model_name, dataset_name, hpo_name, top_n):
    if is_baseline(hpo_name):
        return ""
    hpo_name1, hpo_name2 = hpo_name.replace("-BTT", ""), hpo_name
    sqlite_path_list1 = get_sqlite_path_list(model_name, dataset_name, hpo_name1)
    sqlite_path_list2 = get_sqlite_path_list(model_name, dataset_name, hpo_name2)

    path_list_dict = {}
    for sqlite_path in sqlite_path_list1 + sqlite_path_list2:
        path_list_dict[sqlite_path] = list(get_specific_final_metric_dict(sqlite_path).values())  ## no top_n

    r_list = []
    for sqlite_path1 in sqlite_path_list1:
        for sqlite_path2 in sqlite_path_list2:
            single_trial_metric_list1 = path_list_dict[sqlite_path1]
            single_trial_metric_list2 = path_list_dict[sqlite_path2]
            metric_list = sorted(single_trial_metric_list1 + single_trial_metric_list2)[:top_n]
            count = 0
            for metric in metric_list:
                if metric in single_trial_metric_list2:
                    count += 1
            r = count / top_n
            r_list.append(r)
    r = np.mean(r_list).round(3)
    print("calc_tophitrate:", r)
    return r


def calc_tsba(model_name, dataset_name, hpo_name):
    if is_baseline(hpo_name):
        return ""
    hpo_name1, hpo_name2 = hpo_name.replace("-BTT", ""), hpo_name
    sqlite_path_list1 = get_sqlite_path_list(model_name, dataset_name, hpo_name1)
    sqlite_path_list2 = get_sqlite_path_list(model_name, dataset_name, hpo_name2)
    return ""


def calc_trial_num(model_name, dataset_name, hpo_name):
    sqlite_path_list = get_sqlite_path_list(model_name, dataset_name, hpo_name)
    trial_num_list = []
    for sqlite_path in sqlite_path_list:
        lst = get_specific_final_metric_dict(sqlite_path)
        trial_num_list.append(len(lst))
    print(trial_num_list)  # ???
    r = int(np.mean(trial_num_list))
    std = int(np.std(trial_num_list))
    s = "+-".join([str(r), str(std)])
    print("calc_trial_num:", s)
    return s


def calc_relation(model_name, dataset_name, hpo_name, loss="train_loss", top_n=None):
    sqlite_path_list = get_sqlite_path_list(model_name, dataset_name, hpo_name)
    e_list = []
    for sqlite_path in sqlite_path_list:
        test_loss_dict = get_specific_final_metric_dict(sqlite_path)
        if top_n is not None:
            test_loss_dict = dict(sorted(test_loss_dict.items(), key=lambda x: x[1])[:top_n])
        id_list = list(set(test_loss_dict.keys()))
        test_loss_list = [test_loss_dict[id] for id in id_list]

        _loss_dict = get_specific_last_period_metric_dict(sqlite_path, key=loss)
        _loss_list = [_loss_dict[id] for id in id_list]
        e_list.append(np.corrcoef(test_loss_list, _loss_list)[0][1])
    e = np.mean(e_list).round(1)
    print("calc_relation:", e)
    return e


def is_baseline(hpo_name):
    return hpo_name in ["Random", "SMAC"]


if __name__ == '__main__':
    # model_name_list = ["TimesNet", "Transformer"]
    # dataset_name_list = ["ETTh1", "Traffic"]
    # hpo_name_list = ["Random", "SMAC", "Random-BTT", "SMAC-BTT"]
    # Transformer	ETTh1	SMAC

    # Transformer	ETTh1	Random-BTT

    model_name_list = ["Transformer"]
    dataset_name_list = ["ETTh1"]
    hpo_name_list = ["Random-BTT", "SMAC-BTT"]

    for model_name in model_name_list:
        for dataset_name in dataset_name_list:
            for hpo_name in hpo_name_list:
                print(model_name, dataset_name, hpo_name)
                r = list()
                r.append(calc_top(model_name, dataset_name, hpo_name, 1, "best"))
                r.append(calc_top(model_name, dataset_name, hpo_name, 1, "avg"))
                r.append(calc_top(model_name, dataset_name, hpo_name, 10, "avg"))

                r.append(calc_tophitrate(model_name, dataset_name, hpo_name, 10))
                r.append(calc_tsba(model_name, dataset_name, hpo_name))

                r.append(calc_trial_num(model_name, dataset_name, hpo_name))

                r.append(calc_relation(model_name, dataset_name, hpo_name, "train_loss", None))
                r.append(calc_relation(model_name, dataset_name, hpo_name, "val_loss", None))
                r.append(calc_relation(model_name, dataset_name, hpo_name, "train_loss", 10))
                r.append(calc_relation(model_name, dataset_name, hpo_name, "val_loss", 10))
                s = "\t".join([str(i) for i in r])
                print(s)
                print()
