import os
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
    return d


def get_periodical_data(sqlite_path=None):
    conn = sqlite3.connect(sqlite_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM MetricData WHERE type='PERIODICAL'")
    data = cursor.fetchall()
    return data


def get_ideal_data(sqlite_path):
    txt_path = sqlite_path.replace(".sqlite", ".txt")
    print("get_ideal_data:")
    if os.path.exists(txt_path):
        metric_list = np.loadtxt(txt_path)
    else:
        conn = sqlite3.connect(sqlite_path)
        cur = conn.cursor()
        sql = "SELECT * FROM MetricData WHERE type='PERIODICAL'"
        cur.execute(sql)
        values = cur.fetchall()
        id_metric_dict = {}
        for i in range(len(values)):
            trial_id = values[i][1]
            d = yaml.load(eval(values[i][5]), Loader=yaml.FullLoader)
            val = float(d["default"]) if type(d) == dict else float(d)
            id_metric_dict[trial_id] = val
        metric_list = list(id_metric_dict.values())
        np.savetxt(txt_path, metric_list)
    return metric_list


def simulate(sqlite_path, loss_flag, use_pre=False):
    def trans_train(d):
        if "loss" in d:
            d["train_loss"] = d["loss"]
            d["train_loss_list"] = d["loss_list"]
        if "acc" in d:
            d["train_acc"] = d["acc"]
            d["train_acc_list"] = d["acc_list"]
        return d

    print("simulate:")
    symptom_name_list = ["VG", "EG", "DR", "SC", "HO", "NG"]
    count_dict = {"total_trial": 0, "ill_trial": 0, "VG": 0, "EG": 0, "DR": 0, "SC": 0, "HO": 0, "NG": 0}

    sim_path1 = sqlite_path.replace(".sqlite", "_sim.txt")
    sim_path2 = sqlite_path.replace(".sqlite", "_sim_count.txt")
    if os.path.exists(sim_path1) and use_pre:
        metric_list = np.loadtxt(sim_path1)
        count_dict_value = np.loadtxt(sim_path2)
        for i, k in enumerate(count_dict.keys()):
            count_dict[k] = count_dict_value[i]
        return metric_list, count_dict

    max_epoch = 20
    conf = get_assessor_config(loss_flag)
    print(conf)
    values = get_periodical_data(sqlite_path)
    my_assessor = MyAssessor(**conf)

    metric_list = []
    id_result_dict_list_dict = {}
    et_id_set = set()
    for i in range(len(values)):
        if i % (len(values) / 10) == 0:
            print(i, "/", len(values), end=" ||| ")
        trial_id = values[i][1]
        result_dict = yaml.load(eval(values[i][5]), Loader=yaml.FullLoader)
        result_dict = trans_train(result_dict)
        id_result_dict_list_dict[trial_id].append(result_dict) \
            if trial_id in id_result_dict_list_dict else id_result_dict_list_dict.update({trial_id: [result_dict]})
        result_dict_list = id_result_dict_list_dict[trial_id]
        epoch_idx = len(result_dict_list)
        # print(result_dict)
        if epoch_idx == max_epoch:
            if trial_id in et_id_set:
                et_id_set.remove(trial_id)
            else:
                metric_list.append(result_dict["default"])
            continue
        if trial_id in et_id_set:
            continue

        has_symptom = my_assessor.assess_trial(trial_id, result_dict_list)
        if has_symptom:
            et_id_set.add(trial_id)
            metric_list.append(result_dict["default"])
            count_dict["ill_trial"] += 1
            for symptom_name in symptom_name_list:
                if my_assessor.info_dict["has_symptom"]:
                    count_dict[symptom_name] += 1
                if my_assessor.info_dict[symptom_name] is not None:
                    count_dict[symptom_name] += 1

    count_dict["total_trial"] = len(id_result_dict_list_dict.keys())
    np.savetxt(sim_path1, metric_list)
    np.savetxt(sim_path2, list(count_dict.values()))
    return metric_list, count_dict


def test():
    # sqlite_path = "/Users/admin/Desktop/sqlite_files/cifar10cnn/random_monitor/6u4f23m9.sqlite"
    sqlite_path = "/Users/admin/Desktop/sqlite_files/exchange96auto/random_monitor/mu6i1bnd.sqlite"
    loss_flag = True if "exchange96auto" in sqlite_path else False
    ideal_metric_list = get_ideal_data(sqlite_path)
    simulate_metric_list, count_dict = simulate(sqlite_path, loss_flag, use_pre=False)
    top_k = -1

    lst = [ideal_metric_list, simulate_metric_list]
    c_lst = ["r", "b"]
    plt.figure(figsize=(10, 7.5))
    for idx in range(len(lst)):
        metric_list = lst[idx]
        metric_list = sorted(metric_list, reverse=not loss_flag)
        metric_list = metric_list[:top_k]
        plot_y = metric_list
        plot_x = np.linspace(0, len(metric_list), len(metric_list))
        plt.plot(plot_x, plot_y, label="ideal" if idx == 0 else "simulate", color=c_lst[idx])
        plt.legend()
        plt.xlabel("Sorted Configuration Number")
        if loss_flag:  # log scale
            plt.ylabel("MSE Loss (log scale)")
            plt.yscale("log")
            plt.y_range = (0.1, 1)
        else:
            plt.ylabel("Validation Accuracy")
    plt.show()

    if count_dict is None:
        return
    # draw hist of count_dict
    plt.figure(figsize=(10, 7.5))
    plt.bar(count_dict.keys(), count_dict.values(), color="b")
    # show number on hist
    for a, b in zip(count_dict.keys(), count_dict.values()):
        plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=10)
    plt.show()


if __name__ == '__main__':
    test()
