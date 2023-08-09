import os.path
import pickle
import sqlite3

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.backends.backend_pdf import PdfPages


def plot_rank():
    base_dir = "/Users/admin/Desktop/sqlite_files/cifar10cnn/"
    file_lst = ["random/9q6swc3z", "smac/i71sdwal", "gp/qid4b7ep"]
    label_lst = ["random", "smac", "gp"]
    loss_flag = False

    # base_dir = "/Users/admin/Desktop/sqlite_files/exchange96auto/"
    # file_lst = ["random/julky8q4", "gp/bu7lrjdf", "tpe/0tnp96s3", "gp_msr/vg8blops"]
    # label_lst = ["random", "gp", "tpe", "gp_msr"]
    # loss_flag = True

    plt.figure(figsize=(8.4, 6.3))
    for idx in range(len(file_lst)):
        db_path = os.path.join(base_dir, file_lst[idx])
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        sql = "SELECT * FROM MetricData WHERE type='PERIODICAL'"  ###
        cur.execute(sql)
        values = cur.fetchall()
        print("begin to fetch data x:")
        id_metric_dict = {}
        for i in range(len(values)):
            trial_id = values[i][1]
            d = yaml.load(eval(values[i][5]), Loader=yaml.FullLoader)
            val = float(d["default"]) if type(d) == dict else float(d)
            # val = min(np.log(val), 0.5) if loss_flag else val
            id_metric_dict[trial_id] = val
        metric_list = list(id_metric_dict.values())
        time_rate = 6 / 6  # 默认字典按照时间排序 取前n%的数据 注意：1/6看前期 3/6看中期 6/6看后期
        metric_list = metric_list[:int(len(metric_list) * time_rate)]

        print("label: ", label_lst[idx], "num: ", len(metric_list))
        print("begin to plot data:")

        top_num = 100  # 取排名前n%的数据 -10看整体趋势 +10看最佳配置
        metric_list = np.sort(metric_list)
        if not loss_flag:
            metric_list = metric_list[::-1]
        metric_list = metric_list[:top_num]
        plot_y = metric_list
        plot_x = np.linspace(0, len(metric_list), len(metric_list))
        plt.plot(plot_x, plot_y, label=label_lst[idx])
        plt.legend()
        plt.xlabel("Sorted Configuration Number")
        if loss_flag:  # log scale
            plt.ylabel("MSE Loss (log scale)")
            plt.yscale("log")
            plt.y_range = (0.1, 1)
        else:
            plt.ylabel("Validation Accuracy")
    plt.show()


def plot_phenomenon():
    #     细化特征现象图
    #     1 三张图展示三种类型的特征： 类似distribution
    #         1.1 最终的验证集正确率x (validate_acc) | 早期的训练损失值y (train_loss) -> cifar10cnn
    #         1.2 最终的验证集正确率x (validate_acc) | 早期的梯度中值y (median_abs_grad) -> cifar10lstm
    #         1.3 最终的验证集损失值x  (validate_loss) | 早期的激活覆盖率y (active_layer_ratio) -> traffic96aoto
    #         (early stage) ||| (last epoch)
    use_pre = True
    data_limit = 6000  # good: percent%, bad: percent%

    font_size = 20
    from _simulate import simulate
    sqlite_files_dir = "/Users/admin/Desktop/sqlite_files"

    def set_fig():
        plt.rcParams.update({'font.size': int(font_size * 2 / 3)})  ###
        fig_size_base = [8, 6]
        fig_size = tuple([i for i in fig_size_base])
        plt.figure(figsize=fig_size)

        # return 20 if loss_flag else 20

    def get_metric(result_dict, metric_name):
        if metric_name == "val_acc":
            return result_dict["val_acc"]
        elif metric_name == "val_loss":
            return result_dict["val_loss"]
        elif metric_name == "train_loss":
            return result_dict["train_loss"]
        elif metric_name == "absolute_gradient_value":
            lst = result_dict["weight_grad_abs_avg_1da"]
            if type(lst) is dict:
                lst = np.array(lst["__ndarray__"])
            if type(lst[0]) is not np.float64:
                return None
            return np.median(lst)
        elif metric_name == "active_layer_ratio":
            lst = result_dict["weight_grad_rate0_1da"]
            if type(lst) is dict:
                lst = np.array(lst["__ndarray__"])
            if type(lst[0]) is not np.float64:
                return None
            return 1 - np.count_nonzero(lst) / len(lst)

    def _plot_early_feature_distribution(early_metric_name, final_metric_name, scene_name, early_idx, percent, density):
        loss_flag = True if "96" in scene_name else False

        file_name = \
            [i for i in os.listdir(os.path.join(sqlite_files_dir, scene_name, "_monitor")) if i.endswith(".sqlite")][0]
        sqlite_path = os.path.join(sqlite_files_dir, scene_name, "_monitor", file_name)
        raw_metric_list, our_metric_list, count_dict, not_ill_metric_list, raw_id_metric_list_dict, our_id_epoch_dict, \
        not_ill_id_epoch_dict, symptom_metric_list_dict, ill_metric_list, healthy_metric_list, id_result_dict_list_dict, \
        id_timestamp_list_dict, symptom_id_list_dict = simulate(sqlite_path, loss_flag, data_limit, use_pre)
        bin_num = 20
        interval = 2 if bin_num > 5 else 1
        if loss_flag:
            good_split_val = np.percentile(raw_metric_list, percent)
            bad_split_val = np.percentile(raw_metric_list, 100 - percent)
        else:
            good_split_val = np.percentile(raw_metric_list, 100 - percent)
            bad_split_val = np.percentile(raw_metric_list, percent)
        good_val_list, bad_val_list = [], []
        for _id in id_result_dict_list_dict.keys():
            final_metric = get_metric(id_result_dict_list_dict[_id][-1], final_metric_name)
            early_metric = get_metric(id_result_dict_list_dict[_id][early_idx], early_metric_name)
            print(early_metric, final_metric)
            if type(final_metric) is not float or type(early_metric) is not float:
                continue
            if loss_flag:
                if final_metric <= good_split_val:
                    good_val_list.append(early_metric)
                if final_metric >= bad_split_val:
                    bad_val_list.append(early_metric)
            else:
                if final_metric >= good_split_val:
                    good_val_list.append(early_metric)
                if final_metric <= bad_split_val:
                    bad_val_list.append(early_metric)
        set_fig()
        print("good: {}, bad: {}".format(len(good_val_list), len(bad_val_list)))
        # good_label = "Good Trial (Top {}%)".format(percent)
        # bad_label = "Bad Trial (Bottom {}%)".format(percent)
        # Top 10% performance
        good_label = "Top {}% performance".format(percent)
        bad_label = "Bottom {}% performance".format(percent)
        lst = good_val_list + bad_val_list
        val_min, val_max = min(lst), max(lst)
        y_good, _ = np.histogram(good_val_list, bins=bin_num, range=(val_min, val_max), density=density)
        y_bad, _ = np.histogram(bad_val_list, bins=bin_num, range=(val_min, val_max), density=density)
        x = np.arange(bin_num)
        y_all = y_good + y_bad
        # plt.bar(x, y_bad, color="r", alpha=0.5, label=bad_label)
        # plt.bar(x, y_good, color="g", alpha=0.5, label=good_label)
        plt.bar(x, y_all, color="b", label=bad_label)
        plt.bar(x, y_good, color="r", label=good_label)
        xticks = np.linspace(val_min, val_max, bin_num).round(2)
        xticks = [str(xticks[i]) if i % interval == 0 else "" for i in range(bin_num)]
        plt.xticks(x, xticks)
        plt.xlabel("{}".format(early_metric_name.replace("_", " ")), fontsize=font_size)
        y_label = " ".join(["Density" if density else "Number of trials"])
        plt.ylabel(y_label, fontsize=font_size)
        plt.legend(fontsize=int(font_size * 2 / 3))
        scene_title_dict = {"cifar10cnn": "Cifar10CNN", "cifar10lstm": "Cifar10LSTM",
                            "exchange96auto": "Exchange96Transformer"}
        scene_title = scene_title_dict[scene_name]
        plt.title(scene_title, fontsize=font_size)
        fig_path = os.path.join("figs", "early", early_metric_name + ".png")
        plt.savefig(fig_path)
        fig_path_list = fig_path.split("/")
        pdf_path = "/".join(["/".join(fig_path_list[:-1]), "pdf", fig_path_list[-1].replace(".png", ".pdf")])
        pdf = PdfPages(pdf_path)
        pdf.savefig()
        pdf.close()
        plt.show()

    def plot_p1():
        early_idx = 5
        percent = 10
        density = True
        scene_name = "cifar10cnn"
        final_metric_name = "val_acc"
        early_metric_name = "absolute_gradient_value"  # "train_loss"
        _plot_early_feature_distribution(early_metric_name, final_metric_name, scene_name, early_idx, percent, density)

    def plot_p2():
        early_idx = 15
        percent = 30
        density = False
        # scene_name = "cifar10lstm"
        scene_name = "exchange96auto"  # exchange96auto
        final_metric_name = "val_loss"
        early_metric_name = "active_layer_ratio"
        _plot_early_feature_distribution(early_metric_name, final_metric_name, scene_name, early_idx, percent, density)

    def plot_p3():
        early_idx = 5
        percent = 10
        density = True
        scene_name = "cifar10lstm"
        final_metric_name = "val_acc"
        early_metric_name = "train_loss"
        _plot_early_feature_distribution(early_metric_name, final_metric_name, scene_name, early_idx, percent, density)

    # plot_p1()
    plot_p2()
    # plot_p3()
    pass


def plot_time():
    # label_lst = ["random", "random_lce", "gp", "gp_lce", "tpe", "tpe_lce", "smac", "smac_lce"]  # 1 -> n
    # label_lst = ["random", "random_msr", "gp", "gp_msr", "tpe", "tpe_msr","smac","smac_msr"]  # 1 -> n

    scene_name_list = ["cifar10cnn", "cifar10lstm", "exchange96auto"]  # "traffic96trans"
    scene_title_list = ["Cifar10CNN", "Cifar10LSTM", "Ex96Trans"]
    scene_title_dict = {"cifar10cnn": "Cifar10CNN", "cifar10lstm": "Cifar10LSTM",
                        "exchange96auto": "Ex96Trans"}

    hpo_name_lst = ["random", "gp", "tpe", "smac"]
    etr_name_list = ["","our_"]  # "", "lce_", "msr_", "our_"
    color_dict = {"random": "b", "gp": "g", "tpe": "y", "smac": "c"}
    line_style_dict = {"": 'dashdot', "lce": ':', "msr": ':', "our": "solid"}
    top_k_list = [1, 3, 10]  # [1, 3, 5, 10]

    seg_num_list1 = [120, 120, 30, 30]  # acc
    seg_num_list2 = [30, 30, 12, 6]  # loss
    time_rate_list = [6 / 6, 6 / 6, 6 / 6, 2 / 6]
    start_seg_rate_list = [0 / 6, 0 / 6, 1 / 3 / 6, 0 / 6]


    etr_name_list = ["","our_"]  # "", "lce_", "msr_", "our_"


    # cifar10cnn: bad smac
    # cifar10lstm: bad all
    # exchange96auto: ok all
    # traffic96trans: gp ok / bad all
    # scene_name_list = ["exchange96auto"]
    # top_k_list = [10]
    # start_seg_rate_list = [1/3 / 6]
    # time_rate_list = [6 / 6, 6 / 6, 6 / 6, 2 / 6]

    # seg_num_list1 = [60] * 4
    # seg_num_list2 = seg_num_list1

    # etr_hpo_name_lst = [prefix + name for prefix in etr_name_list for name in hpo_name_lst]

    table_flag = False

    fontsize = 25

    def set_fig():
        plt.rcParams.update({'font.size': int(fontsize * 2 / 3)})  ###
        fig_size_base = [8, 6]
        fig_size = tuple([i * 1.5 for i in fig_size_base])
        plt.figure(figsize=fig_size)

    def end_fig():
        plt.title(scene_title, fontsize=fontsize)
        plt.xlabel("Time (hour)", fontsize=fontsize)
        if loss_flag:  # log scale ?
            plt.ylabel("MSE " + " (top{})".format(top_k), fontsize=fontsize)
            # plt.yscale("log")
        else:
            plt.ylabel("Classification Accuracy " + " (top{})".format(top_k), fontsize=fontsize)
        plt.legend(fontsize=int(fontsize * 2 / 3))  ###
        fig_path = os.path.join("figs", "time", "top" + str(top_k), scene_name + ".png")
        plt.savefig(fig_path)
        pdf_path = os.path.join("figs", "time", "top" + str(top_k), "pdf", scene_name + ".pdf")
        pdf = PdfPages(pdf_path)
        pdf.savefig()
        pdf.close()
        if not table_flag:
            plt.show()

    def _plot_time(scene_name, hpo_name, top_k, seg_num, time_rate, start_seg_rate, table_flag):
        # base_dir = os.path.join("/Users/admin/Desktop/sqlite_files/", scene_name)
        base_dir = "/Users/admin/Desktop/清华软院/2023春/1-清软毕设BTT/实验结果数据/sqlite_files/"
        base_dir = os.path.join(base_dir, scene_name)
        etr_hpo_name_lst = [etr_name + hpo_name for etr_name in etr_name_list]
        for i in range(len(etr_hpo_name_lst)):
            etr_hpo_name = etr_hpo_name_lst[i]
            hpo_dir = os.path.join(base_dir, etr_hpo_name)
            if not os.path.exists(hpo_dir):
                continue
            if not table_flag:
                print(hpo_dir, os.listdir(hpo_dir))
            file_name_list = [f_name for f_name in os.listdir(hpo_dir) if f_name.endswith(".sqlite")]
            if len(file_name_list) == 0:
                continue
            file_num = len(file_name_list)
            plot_y_2da = np.zeros((file_num, seg_num))  # axis1:file, axis2: metric
            count_y_2da = np.zeros((file_num, seg_num))
            for file_idx in range(file_num):
                sqlite_path = os.path.join(base_dir, etr_hpo_name, file_name_list[file_idx])
                pkl_path = sqlite_path.replace(".sqlite", "_metric_list.pkl")

                if os.path.exists(pkl_path):
                    metric_list = pickle.load(open(pkl_path, "rb"))
                else:
                    from _simulate import get_default_metric_list
                    metric_list = get_default_metric_list(sqlite_path, None)
                    pickle.dump(metric_list, open(pkl_path, "wb"))
                metric_list = metric_list[:int(len(metric_list) * time_rate)]
                count_y = np.array(
                    [len(metric_list[:int(len(metric_list) * ((i + 1) / seg_num))]) for i in range(seg_num)])
                plot_y = np.array([np.mean(np.sort(metric_list[:int(len(metric_list) * ((i + 1) / seg_num))])[:top_k])
                                   if loss_flag else
                                   np.mean(
                                       np.sort(metric_list[:int(len(metric_list) * ((i + 1) / seg_num))])[::-1][:top_k])
                                   for i in range(seg_num)])
                plot_y_2da[file_idx] = plot_y
                count_y_2da[file_idx] = count_y
            plot_y = np.mean(plot_y_2da, axis=0)
            count_y = np.mean(count_y_2da, axis=0)

            for ii in range(len(plot_y) - 1):
                if (loss_flag and plot_y[ii + 1] > plot_y[ii]) or (not loss_flag and plot_y[ii + 1] < plot_y[ii]):
                    plot_y[:ii + 1] = plot_y[ii + 1]

            tmp_start_seg = int(seg_num * start_seg_rate)
            plot_y = plot_y[tmp_start_seg:]
            max_time = 6 * time_rate

            # delta_list = (list((plot_y[:-1] - plot_y[1:]) >= 0) if loss_flag else list((plot_y[1:] - plot_y[:-1]) <= 0))
            tmp_start_seg = tmp_start_seg
            shift_rate = 1 / 3 / 6
            plot_x = np.linspace((start_seg_rate + shift_rate) * max_time, max_time, seg_num - tmp_start_seg)

            line_style = line_style_dict[""] if "_" not in etr_hpo_name else None
            line_style = line_style_dict[etr_hpo_name.split("_")[-2]] if line_style is None else line_style
            color = color_dict[etr_hpo_name.split("_")[-1]]
            d = {"random": "Random", "gp": "GP", "tpe": "TPE", "smac": "SMAC"}
            if "our" in etr_hpo_name:  ####
                # our_gp - > GP-BTTackler
                label = d[hpo_name] + "-BTTackler"
                plt.plot(plot_x, plot_y, label=label, linestyle=line_style, color=color, linewidth=3)
            elif "_" in etr_hpo_name:
                label = d[hpo_name] + "-" + etr_hpo_name.split("_")[0]
                plt.plot(plot_x, plot_y, label=label, linestyle=line_style, color=color, linewidth=2)
            else:
                plt.plot(plot_x, plot_y, label=d[hpo_name], linestyle=line_style, color=color, linewidth=2)
            # 1h 3h 6h /6 -> seg=12
            y_res = [plot_y[-1]]
            s = " & ".join([str(round(i, 4)) for i in y_res])
            count_res = [count_y[seg_num // 6], count_y[len(count_y) // 2], count_y[-1]]
            ss = " & ".join([str(int(i)) for i in count_res])
            print("|".join([scene_name, etr_hpo_name, "(top{})".format(top_k), ss, s]))
            # print()
        print()

    for scene_name, scene_title, time_rate, start_seg_rate in zip(scene_name_list, scene_title_list, time_rate_list,
                                                                  start_seg_rate_list):
        loss_flag = True if "96" in scene_name else False
        for top_k, seg_num1, seg_num2 in zip(top_k_list, seg_num_list1, seg_num_list2):
            set_fig()
            for hpo_name in hpo_name_lst:
                _plot_time(scene_name, hpo_name, top_k, seg_num2 if loss_flag else seg_num1, time_rate, start_seg_rate,
                           table_flag)
            end_fig()


def plot_tmp():
    fontsize = 20
    pkl_path = "/Users/admin/Desktop/sqlite_files/cifar10lstm/random/9412y0ie_metric_list.pkl"
    metric_list = pickle.load(open(pkl_path, "rb"))
    plot_y, _ = np.histogram(metric_list, bins=5)
    plot_y = plot_y / np.sum(plot_y)
    plt.bar(np.arange(len(plot_y)), plot_y)
    plt.bar(np.arange(len(plot_y))[-1], plot_y[-1], color="red")
    plt.ylabel("Number of Trials (ratio)", fontsize=fontsize)
    plt.xlabel("Accuracy", fontsize=fontsize)
    plt.show()


if __name__ == '__main__':
    # plot_rank()

    plot_time()
    # plot_phenomenon()
    # plot_tmp()
