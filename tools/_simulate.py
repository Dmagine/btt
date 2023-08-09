import os
import pickle
import sqlite3
import sys

import numpy as np
import yaml
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.append("../new_package")
from new_package.atdd_assessor import MyAssessor


def get_assessor_config(loss_flag):
    file_path = "../new_package/atdd_default_config.yaml"
    file_obj = open(file_path, 'r')
    info_dict = yaml.load(file_obj, Loader=yaml.FullLoader)
    file_obj.close()
    d = {}
    d.update({"shared": info_dict["shared"]})
    d.update({"basic": info_dict["assessor"]["classArgs"]["basic"]})
    d.update({"compare": info_dict["assessor"]["classArgs"]["compare"]})
    d.update({"diagnose": info_dict["assessor"]["classArgs"]["diagnose"]})
    if loss_flag:
        d["shared"]["enable_dict"]["acc"] = False
    else:
        d["shared"]["enable_dict"]["acc"] = True
    return d


def get_periodical_values(sqlite_path, data_limit):
    conn = sqlite3.connect(sqlite_path)
    cur = conn.cursor()
    sql = "SELECT * FROM MetricData WHERE type='PERIODICAL' limit " + str(data_limit)
    cur.execute(sql)
    values = cur.fetchall()
    values = list(values)
    for i in range(len(values)):
        values[i] = list(values[i])
        d = yaml.load(eval(values[i][5]), Loader=yaml.FullLoader)
        weight_grad_abs_avg_1da = d["weight_grad_abs_avg_1da"]
        if type(weight_grad_abs_avg_1da) is dict:
            d["weight_grad_abs_avg_1da"] = np.array(d["weight_grad_abs_avg_1da"]["__ndarray__"])
            d["weight_grad_rate0_1da"] = np.array(d["weight_grad_rate0_1da"]["__ndarray__"])
        values[i][5] = d
    return values


def get_default_metric_list(sqlite_path, data_limit):
    conn = sqlite3.connect(sqlite_path)
    cur = conn.cursor()
    sql = "SELECT * FROM MetricData WHERE type='PERIODICAL' limit " + str(data_limit) \
        if data_limit is not None else "SELECT * FROM MetricData WHERE type='PERIODICAL'"
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


overview_pie_name_list = ["healthy", "NMG", "ill"]
symptom_pie_name_list = ["EG", "VG", "DR", "SC", "HO", "NMG"]  # "NMG" is "symptom" not "ill"
rule_pie_name_list = ["eg_rule1", "eg_rule2", "eg_rule3",
                      "vg_rule1", "vg_rule2", "vg_rule3", "vg_rule4",
                      "dr_rule1", "dr_rule2", "dr_rule3",
                      "sc_rule1", "sc_rule2", "sc_rule3",
                      "ho_rule1", "ho_rule2",
                      "nmg_rule1", "nmg_rule2"]
symptom_name_list = symptom_pie_name_list
rule_name_list = rule_pie_name_list


def get_metric_ratio(d, metric_name):
    if metric_name in d:
        return d[metric_name] / d["ill_trial"]
    else:
        # overview_list = ["healthy", "ill", "No_more_gain"]
        if metric_name == "healthy":
            return (d["total_trial"] - d["ill_trial"] - d["NMG"]) / d["total_trial"]
        elif metric_name == "ill":
            return d["ill_trial"] / d["total_trial"]
        elif metric_name == "no_more_gain":
            return d["NMG"] / d["total_trial"]


def simulate(sqlite_path, loss_flag, data_limit, use_pre=False):
    def trans_train(d):
        if "loss" in d:
            d["train_loss"] = d["loss"]
            d["train_loss_list"] = d["loss_list"]
        if "acc" in d:
            d["train_acc"] = d["acc"]
            d["train_acc_list"] = d["acc_list"]
        return d

    # count_dict = {}
    # for k in symptom_name_list + rule_name_list:
    #     count_dict[k] = 0
    count_dict = {k: 0 for k in symptom_name_list + rule_name_list + ["total_trial", "ill_trial"]}

    path0 = sqlite_path.replace(".sqlite", ".pkl", 1)
    path1 = sqlite_path.replace(".sqlite", "_raw_metric.pkl")
    path2 = sqlite_path.replace(".sqlite", "_our_metric.pkl")
    path3 = sqlite_path.replace(".sqlite", "_count_dict.pkl")
    path4 = sqlite_path.replace(".sqlite", "_not_ill_metric.pkl")
    path5 = sqlite_path.replace(".sqlite", "_raw_id_metric_list_dict.pkl")
    path6 = sqlite_path.replace(".sqlite", "_our_id_epoch_dict.pkl")
    path7 = sqlite_path.replace(".sqlite", "_not_ill_id_epoch_dict.pkl")
    path8 = sqlite_path.replace(".sqlite", "_symptom_metric_list_dict.pkl")
    path9 = sqlite_path.replace(".sqlite", "_ill_metric_list.pkl")
    path10 = sqlite_path.replace(".sqlite", "_healthy_metric_list.pkl")
    path11 = sqlite_path.replace(".sqlite", "_id_result_dict_list_dict.pkl")
    path12 = sqlite_path.replace(".sqlite", "_id_timestamp_list_dict.pkl")
    path13 = sqlite_path.replace(".sqlite", "_symptom_id_list_dict.pkl")

    max_epoch = 20
    conf = get_assessor_config(loss_flag)
    print("sqlite_path:", sqlite_path)
    print("conf:", conf)
    print("loss_flag:", loss_flag)
    print("data_limit:", data_limit)
    print("use_pre:", use_pre)

    if use_pre and os.path.exists(path1) and os.path.exists(path2) and os.path.exists(path3) and os.path.exists(path4) \
            and os.path.exists(path5) and os.path.exists(path6) and os.path.exists(path7) and os.path.exists(path8) and \
            os.path.exists(path9) and os.path.exists(path10) and os.path.exists(path11) and os.path.exists(path12) and \
            os.path.exists(path13):
        raw_metric_list = pickle.load(open(path1, "rb"))
        our_metric_list = pickle.load(open(path2, "rb"))
        count_dict = pickle.load(open(path3, "rb"))
        not_ill_metric_list = pickle.load(open(path4, "rb"))
        raw_id_metric_list_dict = pickle.load(open(path5, "rb"))
        our_id_epoch_dict = pickle.load(open(path6, "rb"))
        not_ill_id_epoch_dict = pickle.load(open(path7, "rb"))
        symptom_metric_list_dict = pickle.load(open(path8, "rb"))
        ill_metric_list = pickle.load(open(path9, "rb"))
        healthy_metric_list = pickle.load(open(path10, "rb"))
        id_result_dict_list_dict = pickle.load(open(path11, "rb"))
        id_timestamp_list_dict = pickle.load(open(path12, "rb"))
        symptom_id_list_dict = pickle.load(open(path13, "rb"))
        return raw_metric_list, our_metric_list, count_dict, not_ill_metric_list, raw_id_metric_list_dict, \
               our_id_epoch_dict, not_ill_id_epoch_dict, symptom_metric_list_dict, ill_metric_list, healthy_metric_list, \
               id_result_dict_list_dict, id_timestamp_list_dict, symptom_id_list_dict

    if os.path.exists(path0) and (len(pickle.load(open(path0, "rb"))) >= data_limit or loss_flag):
        values = pickle.load(open(path0, "rb"))[:data_limit]
    else:
        values = get_periodical_values(sqlite_path, data_limit)
        pickle.dump(values, open(path0, "wb"))

    my_assessor = MyAssessor(**conf)

    id_result_dict_list_dict = {}
    finished_id_set = set()
    raw_metric_list = []
    our_metric_list = []
    not_ill_metric_list = []
    raw_id_metric_list_dict = {}
    our_id_epoch_dict = {}
    not_ill_id_epoch_dict = {}
    symptom_metric_list_dict = {k: [] for k in symptom_name_list}
    ill_metric_list = []
    healthy_metric_list = []
    id_timestamp_list_dict = {}
    symptom_id_list_dict = {k: [] for k in symptom_name_list}
    for i in range(len(values)):
        if i % int(len(values) / 10) == 0:
            print(i, "/", len(values), end=" ||| ")
        timestamp = values[i][0]
        trial_id = values[i][1]
        tmp = values[i][5]
        result_dict = yaml.load(tmp, Loader=yaml.FullLoader) if type(values[i][5]) == str else tmp  #
        result_dict = trans_train(result_dict)
        # rate0
        id_result_dict_list_dict[trial_id].append(result_dict) \
            if trial_id in id_result_dict_list_dict else id_result_dict_list_dict.update({trial_id: [result_dict]})
        result_dict_list = id_result_dict_list_dict[trial_id]
        id_timestamp_list_dict[trial_id].append(timestamp) \
            if trial_id in id_timestamp_list_dict else id_timestamp_list_dict.update({trial_id: [timestamp]})
        step_idx = len(result_dict_list)
        raw_id_metric_list_dict[trial_id].append(result_dict["default"]) \
            if trial_id in raw_id_metric_list_dict else raw_id_metric_list_dict.update(
            {trial_id: [result_dict["default"]]})
        if step_idx == max_epoch:
            raw_metric_list.append(result_dict["default"])
            if trial_id not in finished_id_set:
                healthy_metric_list.append(result_dict["default"])
        if trial_id in finished_id_set:
            continue
        early_stop = my_assessor.assess_trial(trial_id, result_dict_list)
        if early_stop:
            finished_id_set.add(trial_id)
            ill_flag = False
            for symptom_name in symptom_name_list:
                if my_assessor.info_dict[symptom_name] is not None:
                    ill_flag = True if symptom_name != "NMG" else ill_flag
                    count_dict[symptom_name] += 1
                    symptom_metric_list_dict[symptom_name].append(result_dict["default"])
                    symptom_id_list_dict[symptom_name].append(trial_id)
                    for rule_name in my_assessor.info_dict[symptom_name]:
                        count_dict[rule_name] += 1
            count_dict["ill_trial"] += 1 if ill_flag else 0
            ill_metric_list.append(result_dict["default"]) if ill_flag else None
            # if not ill_flag and my_assessor.info_dict["NMG"] is not None and step_idx == max_epoch:
            if not ill_flag and step_idx == max_epoch:
                not_ill_metric_list.append(result_dict["default"])
                not_ill_id_epoch_dict[trial_id] = step_idx - 1
        else:
            if step_idx == max_epoch:
                not_ill_metric_list.append(result_dict["default"])
                not_ill_id_epoch_dict[trial_id] = step_idx - 1

        if step_idx == max_epoch:
            finished_id_set.add(trial_id)
        if early_stop or step_idx == max_epoch:
            our_metric_list.append(result_dict["default"])
            our_id_epoch_dict[trial_id] = step_idx - 1
    print()
    del_id_list = [k for k, v in id_result_dict_list_dict.items() if len(v) < 20]
    id_result_dict_list_dict = {k: v for k, v in id_result_dict_list_dict.items() if k not in del_id_list}
    raw_id_metric_list_dict = {k: v for k, v in raw_id_metric_list_dict.items() if k not in del_id_list}
    our_id_epoch_dict = {k: v for k, v in our_id_epoch_dict.items() if k not in del_id_list}
    not_ill_id_epoch_dict = {k: v for k, v in not_ill_id_epoch_dict.items() if k not in del_id_list}
    symptom_metric_list_dict = {k: v for k, v in symptom_metric_list_dict.items() if len(v) > 0}
    id_timestamp_list_dict = {k: v for k, v in id_timestamp_list_dict.items() if k not in del_id_list}
    # del_id_list

    count_dict["total_trial"] = len(finished_id_set)
    pickle.dump(raw_metric_list, open(path1, "wb"))
    pickle.dump(our_metric_list, open(path2, "wb"))
    pickle.dump(count_dict, open(path3, "wb"))
    pickle.dump(not_ill_metric_list, open(path4, "wb"))
    pickle.dump(raw_id_metric_list_dict, open(path5, "wb"))
    pickle.dump(our_id_epoch_dict, open(path6, "wb"))
    pickle.dump(not_ill_id_epoch_dict, open(path7, "wb"))
    pickle.dump(symptom_metric_list_dict, open(path8, "wb"))
    pickle.dump(ill_metric_list, open(path9, "wb"))
    pickle.dump(healthy_metric_list, open(path10, "wb"))
    pickle.dump(id_result_dict_list_dict, open(path11, "wb"))
    pickle.dump(id_timestamp_list_dict, open(path12, "wb"))
    pickle.dump(symptom_id_list_dict, open(path13, "wb"))
    return raw_metric_list, our_metric_list, count_dict, not_ill_metric_list, raw_id_metric_list_dict, \
           our_id_epoch_dict, not_ill_id_epoch_dict, symptom_metric_list_dict, ill_metric_list, healthy_metric_list, \
           id_result_dict_list_dict, id_timestamp_list_dict, symptom_id_list_dict


def plot_metric(raw_metric_list, our_metric_list, not_ill_metric_list, scene_name, loss_flag, top_k):
    lst = [raw_metric_list, our_metric_list, not_ill_metric_list]
    c_lst = ["b", "r", "g"]
    label_list = ["raw", "our", "not_ill"]
    fig_size_base = [8, 6]
    fig_size = tuple([i * 1.5 for i in fig_size_base])
    fontsize = 18
    plt.figure(figsize=fig_size)
    for idx in range(len(lst)):
        metric_list = lst[idx]
        metric_list = sorted(metric_list, reverse=not loss_flag)
        metric_list = metric_list[:top_k]
        plot_y = metric_list
        plot_x = np.linspace(0, len(metric_list), len(metric_list))
        print(plot_y)
        plt.plot(plot_x, plot_y, label=label_list[idx], c=c_lst[idx], marker="o", markersize=3)
        # legend和label字体太小
        plt.legend(loc="upper right", fontsize=fontsize)
        plt.xlabel("Sorted Configuration Number", fontsize=fontsize)
        if loss_flag:  # log scale
            plt.ylabel("MSE", fontsize=fontsize)
            plt.yscale("log")
            plt.y_range = (0.1, 1)
        else:
            plt.ylabel("Validation Accuracy")
    # fig_path = "figs/metric/" + scene_name + ".png"
    fig_path = os.path.join("figs", "metric", scene_name + ".png")
    plt.savefig(fig_path)
    plt.show()


def plot_pie(count_dict, overview_pie_name_list, symptom_pie_name_list, rule_pie_name_list, scene_name):
    if count_dict is None:
        return
    column_2d_list = [overview_pie_name_list, symptom_pie_name_list, rule_pie_name_list]
    fig_dir_list = ["overview", "symptom", "rule"]
    title_list = ["Overview", "Symptoms", "Rules"]
    c_list = ["#ff7f0e", "#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
              "#17becf", "#ffbb78", "#aec7e8", "#98df8a", "#ff9896", "#c5b0d5", "#c49c94", "#f7b6d2", "#c7c7c7",
              "#dbdb8d", "#9edae5"]
    fontsize = 20
    plt.rcParams.update({'font.size': fontsize})
    for i in range(len(column_2d_list)):
        column_list = column_2d_list[i]
        d = {k: get_metric_ratio(count_dict, k) for k in column_list}
        plt.figure(figsize=(8, 6))
        # k
        v_list = [v for v in d.values()]
        k_list = [k if v > 0 else "" for k, v in d.items()]
        plt.pie(v_list, labels=k_list, autopct=lambda p: '{:.0f}%'.format(p) if p > 0 else "", colors=c_list)

        plt.title(" ".join([title_list[i], "of", scene_name]), fontsize=fontsize)
        # fixed legend position
        # legend too left
        plt.legend(loc="upper right", fontsize=fontsize, bbox_to_anchor=(1.4, 1.0))
        fig_path = os.path.join("figs", fig_dir_list[i], scene_name + ".png")
        # transform png to pdf
        pdf_path = os.path.join("figs", fig_dir_list[i], "pdf", scene_name + ".pdf")
        pdf = PdfPages(pdf_path)
        pdf.savefig()
        pdf.close()
        plt.savefig(fig_path)
        plt.show()


def plot_triangle(raw_id_metric_list_dict, our_id_epoch_dict, scene_name, loss_flag):
    def set_plot():
        fontsize = 16
        plt.rcParams.update({'font.size': int(fontsize * 2 / 3)})
        fig_size_base = [8, 6]
        fig_size = tuple([i * 1.5 for i in fig_size_base])
        plt.figure(figsize=fig_size)
        plt.xlabel("Epoch", fontsize=fontsize)
        plt.xticks(np.arange(0, 20, 2))
        if loss_flag:  # log scale
            # 设置纵坐标的显示范围
            plt.ylabel("MSE", fontsize=fontsize)
            plt.yscale("log")
            plt.ylim(0.1, 1)
        else:
            plt.ylabel("Validation Accuracy")

    x = np.arange(0, 20)
    start = 30
    end = start + 100

    cnt = 0
    set_plot()
    for trial_id in raw_id_metric_list_dict.keys():
        cnt += 1
        if cnt < start or cnt > end:
            continue
        raw_metric_list = raw_id_metric_list_dict[trial_id]
        if len(raw_metric_list) < 20 or trial_id not in our_id_epoch_dict.keys():
            continue
        # not_ill_epoch = not_ill_epoch_dict[trial_id]
        # if not_ill_epoch == 20 - 1:  #
        #     continue
        plt.plot(x[:len(raw_metric_list)], raw_metric_list, label="raw", c="b", marker="o")
    fig_path = os.path.join("figs", "triangle", scene_name + "_before.png")
    plt.savefig(fig_path)
    plt.show()

    cnt = 0
    set_plot()
    for trial_id in our_id_epoch_dict.keys():  ####
        cnt += 1
        if cnt < start or cnt > end:
            continue
        raw_metric_list = raw_id_metric_list_dict[trial_id]
        if len(raw_metric_list) < 20 or trial_id not in our_id_epoch_dict.keys():
            continue
        our_epoch = our_id_epoch_dict[trial_id]
        # if not_ill_epoch == 20 - 1:  #
        #     continue
        our_id_metric_list = raw_metric_list[:our_epoch + 1]  #
        plt.plot(x[:len(our_id_metric_list)], our_id_metric_list, label="healthy+NMG", c="r", marker="o")
    fig_path = os.path.join("figs", "triangle", scene_name + "_after.png")
    plt.savefig(fig_path)
    plt.show()

    print(np.mean(list(our_id_epoch_dict.values())))
    print(len(raw_id_metric_list_dict.keys()), len(our_id_epoch_dict.keys()))


def plot_distribution(raw_metric_list, healthy_metric_list, not_ill_metric_list, ill_metric_list,
                      symptom_metric_list_dict, scene_name, scene_title, loss_flag):
    # bin_num = 10 if loss_flag else 20
    bin_num = 20 if loss_flag else 20
    interval_num = bin_num // 10
    density = False  # if loss_flag else False
    loss_max = 1.2  ### 3

    fontsize = 30
    plt.rcParams.update({'font.size': int(fontsize * 2 / 3)})
    fig_size_base = [8, 6]
    fig_size = tuple([i * 1.5 for i in fig_size_base])
    plt.figure(figsize=fig_size)
    if loss_flag:
        raw_metric_list = [i if i < loss_max else loss_max for i in raw_metric_list]
        healthy_metric_list = [i if i < loss_max else loss_max for i in healthy_metric_list]
        not_ill_metric_list = [i if i < loss_max else loss_max for i in not_ill_metric_list]
        ill_metric_list = [i if i < loss_max else loss_max for i in ill_metric_list]
        for symptom, metric_list in symptom_metric_list_dict.items():
            metric_list = [i if i < loss_max else loss_max for i in metric_list]
            symptom_metric_list_dict[symptom] = metric_list
    if loss_flag:
        # raw_metric_list = np.log(raw_metric_list).tolist()
        # healthy_metric_list = np.log(healthy_metric_list).tolist()
        # not_ill_metric_list = np.log(not_ill_metric_list).tolist()
        # ill_metric_list = np.log(ill_metric_list).tolist()
        pass
    total_lst = raw_metric_list + healthy_metric_list + ill_metric_list
    val_min, val_max = np.min(total_lst), np.max(total_lst)
    x = np.arange(bin_num)

    y_ill, _ = np.histogram(ill_metric_list, bins=bin_num, range=(val_min, val_max), density=density)
    y_healthy, _ = np.histogram(healthy_metric_list, bins=bin_num, range=(val_min, val_max), density=density)
    y_all, _ = np.histogram(raw_metric_list, bins=bin_num, range=(val_min, val_max), density=density)
    y_not_ill, _ = np.histogram(not_ill_metric_list, bins=bin_num, range=(val_min, val_max), density=density)

    # y_healthy + y_ill
    y_no_nmg = []
    for i in range(bin_num):
        y_no_nmg.append(y_healthy[i] + y_ill[i])
    y_no_nmg = np.array(y_no_nmg)

    if loss_flag:
        plt.bar(x, y_no_nmg, color="b", alpha=1.0, label="Bad Trials")
        plt.bar(x, y_not_ill, color="r", alpha=1.0, label="Good Trials")
        pass
    else:
        # plt.bar(x, y_ill, color="r", alpha=0.5, label="ill")
        # plt.bar(x, y_healthy, color="g", alpha=0.5, label="healthy")
        # pei.......
        plt.bar(x, y_no_nmg, color="b", alpha=1.0, label="Bad Trials")
        plt.bar(x, y_healthy, color="r", alpha=1.0, label="Good Trials")
    y_label = "Density" if density else "Number of Trials"
    plt.ylabel(y_label, fontsize=fontsize)
    x_label = "MSE" if loss_flag else "Classification Accuracy"  # (log scale)?????????
    plt.xlabel(x_label, fontsize=fontsize)
    # 设置xticks间隔
    xticks = np.linspace(val_min, val_max, bin_num).round(2)
    xticks = [str(xticks[i]) if i % interval_num == 0 else "" for i in range(bin_num)]
    plt.xticks(x, xticks)
    plt.legend(fontsize=fontsize)
    plt.title(scene_title, fontsize=fontsize)
    fig_path = os.path.join("figs", "distribution", scene_name + ".png")
    pdf_path = os.path.join("figs", "distribution", "pdf", scene_name + ".pdf")
    pdf = PdfPages(pdf_path)
    pdf.savefig()
    pdf.close()
    plt.savefig(fig_path)
    plt.show()

    # val_max = 1
    color_list = ["r", "g", "b", "c", "m", "y", "k"]
    for symptom, metric_list in symptom_metric_list_dict.items():
        # if symptom == "NMG":
        #     continue
        # plt.hist(metric_list, bins=bin_num, alpha=0.5, label=symptom)
        y_symptom, _ = np.histogram(metric_list, bins=bin_num, range=(val_min, val_max))
        plt.bar(x, y_symptom, alpha=0.3, label=symptom, color=color_list.pop(0))
    xticks = np.linspace(val_min, val_max, bin_num).round(2)
    xticks = [str(xticks[i]) if i % interval_num == 0 else "" for i in range(bin_num)]
    plt.xticks(x, xticks)
    plt.legend()
    fig_path = os.path.join("figs", "_distribution_rule", scene_name + ".png")
    # pdf_path = os.path.join("figs", "_distribution_rule", "pdf", scene_name + ".pdf")
    # pdf = PdfPages(pdf_path)
    # pdf.savefig()
    # pdf.close()
    plt.savefig(fig_path)
    plt.show()


def plot_duration_distribution(raw_metric_list, our_metric_list, raw_id_metric_list_dict, our_id_epoch_dict,
                               not_ill_id_epoch_dict, id_timestamp_list_dict, scene_name, scene_title, loss_flag):
    fontsize = 30
    plt.rcParams.update({'font.size': int(fontsize * 2 / 3)})
    fig_size_base = [8, 6]
    fig_size = tuple([i * 1.5 for i in fig_size_base])

    # hist: x -> metric rank, y -> epoch

    bin_num = 5 if loss_flag else 5  ## 10
    width = 0.4
    shift = width * 1.1
    loss_max = 5

    lst = raw_metric_list + our_metric_list
    val_max, val_min = min(max(lst), loss_max), min(lst)
    plt.figure(figsize=fig_size)

    y_our = [[] for _ in range(bin_num)]
    y_raw = [[] for _ in range(bin_num)]
    for trial_id, epoch in our_id_epoch_dict.items():  # our_id_epoch_dict
        # (0, 'timestamp', 'integer', 0, None, 0)
        start = id_timestamp_list_dict[trial_id][0]

        metric = raw_id_metric_list_dict[trial_id][-1]  ###
        metric = min(metric, loss_max) if loss_flag else metric
        idx = int((metric - val_min) / (val_max - val_min) * bin_num)
        idx = min(max(idx, 0), bin_num - 1)

        our_duration = (id_timestamp_list_dict[trial_id][epoch] - start) / 10 ** 3
        # if trial_id not in not_ill_id_epoch_dict: # count ill+healthy only
        #     y_our[idx].append(our_duration)
        y_our[idx].append(our_duration)

        metric = raw_id_metric_list_dict[trial_id][-1]
        metric = min(metric, loss_max) if loss_flag else metric
        idx = int((metric - val_min) / (val_max - val_min) * bin_num)
        idx = min(max(idx, 0), bin_num - 1)

        raw_duration = (id_timestamp_list_dict[trial_id][-1] - start) / 10 ** 3
        y_raw[idx].append(raw_duration)

        # y_our[idx].append(raw_duration)  # truth !!!!! ...

    p = 20 - 1  # max_epoch-1: duration 只包含了19个epoch的运行时间
    y_our = [np.mean(y_our[i]) if len(y_our[i]) != 0 else 0 for i in range(bin_num)]
    y_raw = [np.mean(y_raw[i]) if len(y_raw[i]) != 0 else 0 for i in range(bin_num)]
    y_our = [y_raw[i] / p if y_our[i] < y_raw[i] / p else y_our[i] for i in range(bin_num)]  # 避免柱子太低洗哦过不明显

    x = np.arange(bin_num)
    plt.bar(x, y_raw, label="Random", width=width, color="b", alpha=0.6)
    plt.bar(x + shift, y_our, label="Random-BTT", width=width, color="r", alpha=0.6)  ###
    plt.xlabel("Final Validation Accuracy", fontsize=fontsize) if not loss_flag \
        else plt.xlabel("Final Validation MSE", fontsize=fontsize)
    plt.ylabel("Average Training Duration (s)", fontsize=fontsize)
    plt.xticks(x + shift / 2, [i for i in np.linspace(val_min, val_max, bin_num).round(2)])
    plt.legend(fontsize=fontsize)
    plt.title(scene_title, fontsize=fontsize)
    fig_path = os.path.join("figs", "epoch", scene_name + ".png")
    pdf_path = os.path.join("figs", "epoch", "pdf", scene_name + ".pdf")
    pdf = PdfPages(pdf_path)
    pdf.savefig()
    pdf.close()
    plt.savefig(fig_path)
    plt.show()


def print_efficient(symptom_id_list_dict, id_timestamp_list_dict, our_id_epoch_dict, scene_name):
    # 节约时间/trial
    symptom_saving_dict = {}
    trial_num = len(our_id_epoch_dict.keys())
    for symptom_name in symptom_name_list:
        symptom_id_list = symptom_id_list_dict[symptom_name]
        if len(symptom_id_list) == 0:
            symptom_saving_dict[symptom_name] = 0
            continue
        saving_list = []
        for trial_id in symptom_id_list:
            if trial_id not in id_timestamp_list_dict:
                continue
            cur = id_timestamp_list_dict[trial_id][our_id_epoch_dict[trial_id]]
            end = id_timestamp_list_dict[trial_id][-1]
            saving = (end - cur) / 10 ** 3
            saving_list.append(saving)
        symptom_saving_dict[symptom_name] = np.sum(saving_list) / trial_num  # trial_num | symptom_id_list
    # avg_total_duration
    avg_total_duration = int(np.mean([(v[-1] - v[0]) / 10 ** 3 for k, v in id_timestamp_list_dict.items()]))
    symptom_saving_dict = {k: round(v, 1) for k, v in symptom_saving_dict.items()}
    our_duration = avg_total_duration - int(np.sum([v for k, v in symptom_saving_dict.items()]))
    # print(scene_name, "avg_total_duration (s):", avg_total_duration, "estimated symptom_saving_time", symptom_saving_dict)
    print("|".join(["scene_name", "avg_total_duration (s)", "saved_duration", "symptom"]))
    print(scene_name, avg_total_duration, our_duration, symptom_saving_dict, sep="|")
    print(" & ".join([str(i) if i != 0 else "-" for i in list(symptom_saving_dict.values())]))
    d = {k: len(v) for k, v in symptom_id_list_dict.items()}
    print(d)
    # cifar10cnn|444|226|{'EG': 0, 'VG': 190.7, 'DR': 35.7, 'SC': 0, 'HO': 0, 'NMG': 0.4}
    # cifar10lstm|481|288|{'EG': 15.5, 'VG': 184.1, 'DR': 85.9, 'SC': 1.6, 'HO': 0.8, 'NMG': 0.2}
    # exchange96auto|1033|219|{'EG': 57.2, 'VG': 60.7, 'DR': 0, 'SC': 42.3, 'HO': 21.0, 'NMG': 38.0}


def _test(scene_idx, data_limit, top_k, use_pre):
    scene_name = scene_name_list[scene_idx]
    scene_title_list = ["Cifar10CNN", "Cifar10LSTM", "Ex96Trans"]
    scene_title = scene_title_list[scene_idx]  #
    file_name = \
        [i for i in os.listdir(os.path.join(sqlite_files_dir, scene_name, "_monitor")) if i.endswith(".sqlite")][0]
    sqlite_path = os.path.join(sqlite_files_dir, scene_name, "_monitor", file_name)
    print(sqlite_path)

    loss_flag = True if "96" in sqlite_path else False
    raw_metric_list, our_metric_list, count_dict, not_ill_metric_list, raw_id_metric_list_dict, our_id_epoch_dict, \
    not_ill_id_epoch_dict, symptom_metric_list_dict, ill_metric_list, healthy_metric_list, id_result_dict_list_dict, \
    id_timestamp_list_dict, symptom_id_list_dict = simulate(sqlite_path, loss_flag, data_limit, use_pre)

    # plot_metric(raw_metric_list, our_metric_list, not_ill_metric_list, scene_name, loss_flag, top_k)
    # plot_triangle(raw_id_metric_list_dict, our_id_epoch_dict, scene_name, loss_flag)

    # plot_pie(count_dict, overview_pie_name_list, symptom_pie_name_list, rule_pie_name_list, scene_name)
    plot_distribution(raw_metric_list, healthy_metric_list, not_ill_metric_list, ill_metric_list,
                      symptom_metric_list_dict, scene_name,scene_title, loss_flag)
    # plot_duration_distribution(raw_metric_list, our_metric_list, raw_id_metric_list_dict, our_id_epoch_dict,
    #                            not_ill_id_epoch_dict, id_timestamp_list_dict, scene_name,scene_title, loss_flag)
    # print_efficient(symptom_id_list_dict, id_timestamp_list_dict, our_id_epoch_dict, scene_name)

    # len1 = len(our_id_epoch_dict)
    # len2 = sum([1 if v < 19 else 0 for k, v in our_id_epoch_dict.items()])
    # print(len1, len2, 0)

    # print("average epoch:", np.mean(list(our_id_epoch_dict.values())))


# sqlite_files_dir = "/Users/admin/Desktop/sqlite_files/"
sqlite_files_dir = "/Users/admin/Desktop/清华软院/2023春/1-清软毕设BTT/实验结果数据/sqlite_files/"

scene_name_list = ["cifar10cnn", "cifar10lstm", "exchange96auto"]  # "traffic96trans","exchange96trans"

def test():
    # sqlite_path = "/Users/admin/Desktop/sqlite_files/cifar10cnn/_monitor/6u4f23m9.sqlite"
    # sqlite_path = "/Users/admin/Desktop/sqlite_files/exchange96auto/_monitor/mu6i1bnd.sqlite"
    # sqlite_path = "/Users/admin/Desktop/sqlite_files/cifar10lstm/_monitor/nphicw6e.sqlite"
    # sqlite_path = "/Users/admin/Desktop/sqlite_files/traffic96trans/_monitor/bmv7wr9x.sqlite"

    use_pre = True
    # data_limit = 6000
    data_limit = 60000
    top_k_list = [-1, -1, -1, -1]  # [50,50,20,20]
    for idx in range(len(scene_name_list)):
        _test(idx, data_limit, top_k_list[idx], use_pre)
        print("=====================================")
    # _test(-1, data_limit, -1, use_pre)


if __name__ == '__main__':
    test()
