import os
import pickle
import sqlite3
import sys

import numpy as np
import yaml
from matplotlib import pyplot as plt

sys.path.append("../new_package")
from new_package.atdd_assessor import MyAssessor


def get_assessor_config(loss_flag):
    file_path = "../new_package/atdd_default_config.yaml"
    file_obj = open(file_path, 'r')
    info_dict = yaml.load(file_obj, Loader=yaml.FullLoader)
    file_obj.close()
    d = {}
    d.update({"shared": info_dict["shared"]})
    d.update({"basic": info_dict["assessor"]["class_args"]["basic"]})
    d.update({"compare": info_dict["assessor"]["class_args"]["compare"]})
    d.update({"diagnose": info_dict["assessor"]["class_args"]["diagnose"]})
    if loss_flag:
        d["shared"]["enable_dict"]["acc"] = False
    else:
        d["shared"]["enable_dict"]["acc"] = True
    return d


def get_periodical_values(sqlite_path, date_limit):
    conn = sqlite3.connect(sqlite_path)
    cur = conn.cursor()
    sql = "SELECT * FROM MetricData WHERE type='PERIODICAL' limit " + str(date_limit)
    cur.execute(sql)
    values = cur.fetchall()
    return values


def get_raw_metric_list(sqlite_path, date_limit):
    conn = sqlite3.connect(sqlite_path)
    cur = conn.cursor()
    sql = "SELECT * FROM MetricData WHERE type='PERIODICAL' limit " + str(date_limit)
    cur.execute(sql)
    values = cur.fetchall()
    id_metric_dict = {}
    for i in range(len(values)):
        trial_id = values[i][1]
        d = yaml.load(eval(values[i][5]), Loader=yaml.FullLoader)
        val = float(d["default"]) if type(d) == dict else float(d)
        id_metric_dict[trial_id] = val
    metric_list = list(id_metric_dict.values())
    return metric_list


overview_list = ["healthy_trial", "ill_trial"]
symptom_name_list = ["VG", "EG", "DR", "SC", "HO", "NG"]
rule_name_list = ["vg_rule1", "vg_rule2", "vg_rule3",
                  "eg_rule1", "eg_rule2", "eg_rule3",
                  "dr_rule1", "dr_rule2",
                  "sc_rule1", "sc_rule2",
                  "ho_rule1", "ho_rule2",
                  "ng_rule1", "ng_rule2"]


# colors = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "cyan", "magenta", "brown", "gray", "olive"]


def simulate(sqlite_path, loss_flag, date_limit, use_pre=False):
    def trans_train(d):
        if "loss" in d:
            d["train_loss"] = d["loss"]
            d["train_loss_list"] = d["loss_list"]
        if "acc" in d:
            d["train_acc"] = d["acc"]
            d["train_acc_list"] = d["acc_list"]
        return d

    count_dict = {}
    for k in overview_list + symptom_name_list + rule_name_list:
        count_dict[k] = 0
    # count_dict = {"total_trial": 0, "top_trial": 0, "ill_trial": 0,
    #               "VG": 0, "EG": 0, "DR": 0, "SC": 0, "HO": 0, "NG": 0}
    path0 = sqlite_path.replace(".sqlite", ".pkl", 1)
    path1 = sqlite_path.replace(".sqlite", "_raw_metric.pkl")
    path2 = sqlite_path.replace(".sqlite", "_our_metric.pkl")
    path3 = sqlite_path.replace(".sqlite", "_count_dict.pkl")

    max_epoch = 20
    conf = get_assessor_config(loss_flag)
    print("sqlite_path:", sqlite_path)
    print("conf:", conf)
    print("loss_flag:", loss_flag)
    print("use_pre:", use_pre)

    if os.path.exists(path1):
        raw_metric_list = pickle.load(open(path1, "rb"))
    else:
        raw_metric_list = get_raw_metric_list(sqlite_path, date_limit)
        pickle.dump(raw_metric_list, open(path1, "wb"))

    if use_pre and os.path.exists(path2) and os.path.exists(path3):
        our_metric_list = pickle.load(open(path2, "rb"))
        count_dict = pickle.load(open(path3, "rb"))
        return raw_metric_list, our_metric_list, count_dict

    if os.path.exists(path0):
        values = pickle.load(open(path0, "rb"))
    else:
        values = get_periodical_values(sqlite_path, date_limit)
        pickle.dump(values, open(path0, "wb"))

    my_assessor = MyAssessor(**conf)

    id_result_dict_list_dict = {}
    finished_id_set = set()
    our_metric_list = []
    for i in range(len(values)):
        if i % (len(values) / 10) == 0:
            print(i, "/", len(values), end=" ||| ")
        trial_id = values[i][1]
        result_dict = yaml.load(eval(values[i][5]), Loader=yaml.FullLoader)
        result_dict = trans_train(result_dict)
        id_result_dict_list_dict[trial_id].append(result_dict) \
            if trial_id in id_result_dict_list_dict else id_result_dict_list_dict.update({trial_id: [result_dict]})
        result_dict_list = id_result_dict_list_dict[trial_id]
        step_idx = len(result_dict_list)
        if trial_id in finished_id_set:
            continue
        symptom_flag = my_assessor.assess_trial(trial_id, result_dict_list)
        if symptom_flag:
            finished_id_set.add(trial_id)
            count_dict["ill_trial"] += 1
            for symptom_name in symptom_name_list:
                if my_assessor.info_dict[symptom_name] is not None:
                    count_dict[symptom_name] += 1
                    for rule_name in my_assessor.info_dict[symptom_name]:
                        count_dict[rule_name] += 1
        if step_idx == max_epoch:
            finished_id_set.add(trial_id)
        if symptom_flag or step_idx == max_epoch:
            our_metric_list.append(result_dict["default"])

    count_dict["total_trial"] = len(finished_id_set)
    count_dict["top_trial"] = len(my_assessor.top_id_set)
    count_dict["healthy_trial"] = count_dict["total_trial"] - count_dict["ill_trial"]
    pickle.dump(our_metric_list, open(path2, "wb"))
    pickle.dump(count_dict, open(path3, "wb"))
    return raw_metric_list, our_metric_list, count_dict


def _test(scene_idx, date_limit, top_k, use_pre):
    scene_name = scene_name_list[scene_idx]
    file_name = \
    [i for i in os.listdir(os.path.join(sqlite_files_dir, scene_name, "_monitor")) if i.endswith(".sqlite")][0]
    sqlite_path = os.path.join(sqlite_files_dir, scene_name, "_monitor", file_name)
    print(sqlite_path)

    loss_flag = True if "96" in sqlite_path else False
    # random_metric_list = get_raw_metric_list(sqlite_path, date_limit, use_pre=use_pre)
    raw_metric_list, our_metric_list, count_dict = simulate(sqlite_path, loss_flag, date_limit, use_pre=use_pre)

    lst = [raw_metric_list, our_metric_list]
    c_lst = ["r", "b"]
    fig_size_base = [8, 6]
    fig_size = tuple([i * 1 for i in fig_size_base])
    plt.figure(figsize=fig_size)
    for idx in range(len(lst)):
        metric_list = lst[idx]
        metric_list = sorted(metric_list, reverse=not loss_flag)
        metric_list = metric_list[:top_k]
        plot_y = metric_list
        plot_x = np.linspace(0, len(metric_list), len(metric_list))
        plt.plot(plot_x, plot_y, label="raw" if idx == 0 else "our", color=c_lst[idx])
        plt.legend()
        plt.xlabel("Sorted Configuration Number")
        if loss_flag:  # log scale
            plt.ylabel("MSE Loss (log scale)")
            plt.yscale("log")
            plt.y_range = (0.1, 1)
        else:
            plt.ylabel("Validation Accuracy")
    # fig_path = "figs/metric/" + scene_name + ".png"
    fig_path = os.path.join("figs", "metric", scene_name + ".png")
    plt.savefig(fig_path)
    plt.show()

    if count_dict is None:
        return
    column_2d_list = [overview_list, symptom_name_list, rule_name_list]
    fig_dir_list = ["overview", "symptom", "rule"]
    title_list = ["Overview", "Symptom", "Rule"]
    for i in range(len(column_2d_list)):
        column_list = column_2d_list[i]
        total_count = count_dict["total_trial"] if fig_dir_list[i] == "overview" else count_dict["ill_trial"]
        d = {k: (count_dict[k] / total_count) for k in column_list}
        plt.figure(figsize=(8, 6))
        d = {k: v for k, v in d.items() if v != 0}
        # 饼状图 图上无问题 但是有图例
        plt.pie(d.values(), labels=d.keys(), autopct='%1.1f%%')
        plt.title(title_list[i])
        plt.legend()
        fig_path = os.path.join("figs", fig_dir_list[i], scene_name + ".png")
        plt.savefig(fig_path)
        plt.show()


sqlite_files_dir = "/Users/admin/Desktop/sqlite_files/"
scene_name_list = ["cifar10cnn", "cifar10lstm", "exchange96auto", "traffic96trans"]


def test():
    # sqlite_path = "/Users/admin/Desktop/sqlite_files/cifar10cnn/_monitor/6u4f23m9.sqlite"
    # sqlite_path = "/Users/admin/Desktop/sqlite_files/exchange96auto/_monitor/mu6i1bnd.sqlite"
    # sqlite_path = "/Users/admin/Desktop/sqlite_files/cifar10lstm/_monitor/nphicw6e.sqlite" # sc ho ng !!!
    # sqlite_path = "/Users/admin/Desktop/sqlite_files/traffic96trans/_monitor/bmv7wr9x.sqlite"

    use_pre = False
    date_limit = 3000
    top_k = 50
    for idx in range(len(scene_name_list)):
        _test(idx, date_limit, top_k, use_pre)
        print("=====================================")
    # _test(0, date_limit, top_k, use_pre)


if __name__ == '__main__':
    test()
