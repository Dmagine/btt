import os.path
import sqlite3

import matplotlib.pyplot as plt
import numpy as np
import yaml


def plot():
    tmpp_file_path = "_graph_in.txt"
    f = open(tmpp_file_path)
    s = f.read().strip()
    f.close()

    num_list = [float(x.strip()) for x in s.split("\n")]
    print(num_list)
    print(len(num_list))

    color_list = ['blue', 'blue', 'blue', 'red', 'red', 'red']
    time_list = ["0.25h", "0.5h", "1h", "2h", "3h", "4h"]
    total_num = len(color_list) * len(time_list)
    print(total_num)
    plt.figure(figsize=(16, 8))
    for i in range(len(color_list)):
        x = time_list.copy()
        x.insert(0, "0h")
        y = num_list[i * len(time_list):(i + 1) * len(time_list)]
        y.insert(0, 0)
        plt.plot(x, y, color=color_list[i], marker='o')
    plt.show()


def bar1():
    import matplotlib.pyplot as plt
    tmpp_file_path = "_graph_in.txt"
    f = open(tmpp_file_path)
    s = f.read().strip()
    f.close()

    num_list = [float(x.strip()[0:x.strip().index('±')]) for x in s.split("\n")]  ####
    print(num_list)
    print(len(num_list))

    # type_name_lst = ["random", "tpe", "anneal"]
    # type_name_lst = ["large", "small"] # ["many","few"]
    type_name_lst = ["single", "double", "all"]

    # cmp_name_lst = ["base", "atdd"]
    cmp_name_lst = ["base", "assessor", "inspector", "tuner"]  # double 也类似 all重复也凑合
    top_n = 3
    base_lst = []
    atdd_lst = []
    fig = plt.figure(figsize=(6, 3))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    x = 0
    for i in range(len(num_list)):
        color = 'r' if i % (2 * top_n) >= top_n else 'b'
        x += 0.1 if (i % (top_n * 2) == 0 and i != 0) else 0.06
        ax.bar(x, num_list[i], color=color, width=0.05)

    #
    # ax.bar(X + 0.00, base_lst, color='b', width=0.3)
    # ax.bar(X + 0.3, atdd_lst, color='r', width=0.3)
    # plt.legend()
    plt.show()

    pass


def pie():
    tmpp_file_path = "_graph_in.txt"
    f = open(tmpp_file_path)
    s = f.read().strip()
    f.close()

    data_list = [x.strip() for x in s.split("\n")]
    data_list = [int(x) if x.isdecimal() else x for x in data_list]
    print(data_list)
    print(len(data_list))

    data_dict = {}
    for i in range(len(data_list)):
        it = data_list[i]
        if type(it) is int:
            data_dict[data_list[i - 1]] = it
    print(data_dict)

    plt.figure(figsize=(40, 10))
    plt.subplot(1, 4, 1)
    name_lst = ["VG", "EG", "DR", "OL", "SC"]
    value_lst = [data_dict[k] for k in name_lst]
    name_lst.append("at_else")
    value_lst.append(data_dict["all"] - sum(value_lst))
    plt.pie(value_lst, labels=name_lst)
    plt.legend()

    plt.subplot(1, 4, 2)
    name_lst = ["ExplodingTensor", "UnchangedWeight", "LossNotDecreasing", "AccuracyNotIncreasing", "VanishingGradient"]
    value_lst = [data_dict[k] for k in name_lst]
    name_lst.append("dd_else")
    value_lst.append(data_dict["all"] - sum(value_lst))
    plt.pie(value_lst, labels=name_lst)
    plt.legend()

    plt.subplot(1, 4, 3)
    name_lst = ["loss_nan", "loss_weak"]
    value_lst = [data_dict[k] for k in name_lst]
    name_lst.append("else")
    value_lst.append(data_dict["all"] - sum(value_lst))
    plt.pie(value_lst, labels=name_lst)
    plt.legend()

    plt.subplot(1, 4, 4)
    name_lst = ["sc_nan", "sc_weak"]
    value_lst = [data_dict[k] for k in name_lst]
    name_lst.append("else")
    value_lst.append(data_dict["all"] - sum(value_lst))
    plt.pie(value_lst, labels=name_lst)
    plt.legend()

    # 绘制饼图
    plt.show()
    pass


def bar2():
    import numpy as np
    import matplotlib.pyplot as plt
    tmpp_file_path = "_graph_in.txt"
    f = open(tmpp_file_path)
    s = f.read().strip()
    f.close()

    num_list = [float(x.strip()) for x in s.split("\n")]
    print(num_list)
    print(len(num_list))

    fig = plt.figure(figsize=(6, 3))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    top_n = 10
    name_lst = ["base", "atdd"]
    color_lst = ['b', 'r']
    for i in range(len(name_lst)):
        X = np.arange(top_n)
        ax.bar(X + i * 0.3, num_list[i * top_n:(i + 1) * top_n], color=color_lst[i], width=0.3)
    plt.legend()
    plt.show()

    pass


def bar3():
    import matplotlib.pyplot as plt
    tmpp_file_path = "_graph_in.txt"
    f = open(tmpp_file_path)
    s = f.read().strip()
    f.close()

    num_list = [float(x.strip()[0:x.strip().index('±')]) for x in s.split("\n")]  ####
    print(num_list)
    print(len(num_list))

    type_name_lst = ["single", "double", "all"]
    cmp_name_lst = ["base", "assessor", "inspector", "tuner"]  # double 也类似 all重复也凑合
    top_n = 3
    base_lst = []
    atdd_lst = []
    fig = plt.figure(figsize=(6, 3))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    x = 0
    for i in range(len(num_list)):
        color = 'r' if i % (4 * top_n) >= top_n else 'b'
        x += 0.2 if (i % (top_n * 4) == 0 and i != 0) else 0.1 if (i % top_n == 0) else 0.07
        ax.bar(x, num_list[i], color=color, width=0.05)
    plt.show()

    pass


def plot_reproduce():
    old_id = "rfi4zmp0"  # local:216t07aj remote:rfi4zmp0
    new_id = "pu5jzwdx"  # local:i82b63wc remote:pu5jzwdx

    nni_dir = "~/nni-experiments/"
    desk_dir = "/Users/admin/Desktop/"

    # db_path_old = os.path.join(nni_dir, old_id, "db/nni.sqlite")
    db_path_old = os.path.join(desk_dir, old_id, "nni_old.sqlite")
    print(db_path_old)
    db_path_oldd = os.path.join("./", "nni_old.sqlite")
    os.system(" ".join(["cp", db_path_old, db_path_oldd]))
    conn_old = sqlite3.connect(db_path_oldd)
    cur_old = conn_old.cursor()

    # db_path_new = os.path.join(nni_dir, new_id, "db/nni.sqlite")
    db_path_new = os.path.join(desk_dir, new_id, "nni_new.sqlite")
    db_path_neww = os.path.join("./", "nni_new.sqlite")
    os.system(" ".join(["cp", db_path_new, db_path_neww]))
    conn_new = sqlite3.connect(db_path_neww)
    cur_new = conn_new.cursor()

    sql = "SELECT * FROM MetricData"
    cur_old.execute(sql)
    values = cur_old.fetchall()

    param_id_max_step_dict = {}
    for i in range(len(values)):  # final ??? seq include 0
        param_id = values[i][2]
        param_id_max_step_dict[param_id] = values[i][4] if param_id not in param_id_max_step_dict else \
            max(param_id_max_step_dict[param_id], values[i][4])

    sql = "SELECT * FROM MetricData"
    cur_new.execute(sql)
    values = cur_new.fetchall()

    param_id_metric_list_dict = {}
    max_len = 0
    for i in range(len(values)):  # final ??? seq include 0
        param_id = values[i][2]
        # new -> raw -> float
        # metric = yaml.load(eval(values[i][5]), Loader=yaml.FullLoader)["default"]
        metric = yaml.load(eval(values[i][5]), Loader=yaml.FullLoader)
        if param_id not in param_id_metric_list_dict:
            param_id_metric_list_dict[param_id] = [metric]
        else:
            param_id_metric_list_dict[param_id].append(metric)
            max_len = max(max_len, len(param_id_metric_list_dict[param_id]))
    for key in param_id_metric_list_dict.keys():
        param_id_metric_list_dict[key].pop(-1)
    max_len -= 1

    print(param_id_max_step_dict)
    print(param_id_metric_list_dict)
    print(max_len)

    plt.figure(figsize=(20, 12))
    x = [i for i in range(1, max_len + 1)]
    for key in param_id_max_step_dict.keys():
        split_idx = param_id_max_step_dict[key]
        metric_list = param_id_metric_list_dict[key]
        y1 = metric_list
        x1 = x[:len(y1)]

        y2 = metric_list[:split_idx]
        x2 = x[:len(y2)]

        plt.plot(x1, y1, color='b', marker='o')
        plt.plot(x2, y2, color='r', marker='o')
    plt.show()


def plot_feature():
    beta1 = 0.001
    beta3 = 70
    delta = - 0.01  ##### okkk
    gamma = 0.7
    zeta = 0.03  #####

    def get_veg_metric(param_grad_abs_ave_list, module_name_flow_2dlist, module_name_list):
        # 越小越好
        if 0 in param_grad_abs_ave_list or 'NaN' in param_grad_abs_ave_list:
            return np.log10(beta3) - np.log10(beta1)
        else:
            mid = (np.log10(beta3) - np.log10(beta1))
            lst = []
            for module_name_flow_list in module_name_flow_2dlist:
                module_idx_flow_list = [module_name_list.index(name) for name in module_name_flow_list]
                for i in range(len(module_idx_flow_list) - 1):
                    idx_1 = module_idx_flow_list[i]
                    idx_2 = module_idx_flow_list[i + 1]
                    v1, v2 = param_grad_abs_ave_list[idx_1], param_grad_abs_ave_list[idx_2]
                    val = abs(np.log10(v1 / v2) - mid)

                    max_l = 30
                    for level in range(max_l):
                        b = (np.log10(beta3) - mid) / (2**level)
                        if val > b:
                            lst.append(i - level)
                    lst.append(- max_l - 1)
            return sum(lst) / len(lst)

    def get_ol_metric(acc_list):
        count = 0
        maximum_list = []
        minimum_list = []
        for i in range(len(acc_list)):
            if i == 0 or i == len(acc_list) - 1:
                continue
            if acc_list[i] - acc_list[i - 1] >= 0 and acc_list[i] - acc_list[i + 1] >= 0:
                maximum_list.append(acc_list[i])
            if acc_list[i] - acc_list[i - 1] <= 0 and acc_list[i] - acc_list[i + 1] <= 0:
                minimum_list.append(acc_list[i])
        for i in range(min(len(maximum_list), len(minimum_list))):
            if maximum_list[i] - minimum_list[i] >= zeta:
                count += 1
        return count / len(acc_list)

    def get_sc_metric(acc_list):
        count = 0  # 越多越好
        if len(acc_list) > 1:
            for i in range(1, len(acc_list)):  # !!!!! 0- -1
                if acc_list[i] - acc_list[i - 1] > delta:
                    count += 1
            return count / (len(acc_list) - 1)
        else:
            return 1

    def get_dr_metric(param_grad_zero_rate):
        # 越低越好
        max_l = 7
        for level in range(max_l):
            g = gamma / (10 ** level)
            if param_grad_zero_rate > g:
                return - level
        return - max_l - 1

    old_id = "rfi4zmp0"  # local:216t07aj remote:rfi4zmp0

    desk_dir = "/Users/admin/Desktop/"

    # db_path_old = os.path.join(nni_dir, old_id, "db/nni.sqlite")
    db_path_old = os.path.join(desk_dir, old_id, "nni_old.sqlite")
    print(db_path_old)
    db_path_oldd = os.path.join("./", "nni_old.sqlite")
    os.system(" ".join(["cp", db_path_old, db_path_oldd]))
    conn_old = sqlite3.connect(db_path_oldd)
    cur_old = conn_old.cursor()

    sql = "SELECT * FROM MetricData WHERE type='PERIODICAL'"  # WHERE type='FINAL' 'PERIODICAL'
    cur_old.execute(sql)
    values = cur_old.fetchall()

    id_x_dict = {}  # default
    id_y_dict = {}
    for i in range(len(values)):  # final ??? seq include 0
        param_id = values[i][2]
        d = yaml.load(eval(values[i][5]), Loader=yaml.FullLoader)
        metric_x = d["default"]
        # print(d.keys())
        # final # dict_keys(['default', 'acc', 'loss', 'reward', 'val_acc', 'val_loss', 'val_reward', 'step_counter', 'test_acc', 'test_loss', 'test_reward', 'data_cond', 'weight_cond', 'lr_cond', 'continuous_vg_count', 'continuous_eg_count', 'continuous_dr_count', 'at_symptom', 'dd_symptom', 'wd_symptom', 'cmp_val_loss', 'cmp_sc_metric', 'out_dict_str', 'assessor_stop', 'inspector_stop'])
        # periodical # dict_keys(['default', 'acc', 'loss', 'reward', 'val_acc', 'val_loss', 'val_reward', 'step_counter', 'param_has_inf', 'param_grad_zero_rate', 'data_cond', 'weight_cond', 'lr_cond', 'param_val_var_list', 'param_grad_abs_ave_list', 'module_name_flow_2dlist', 'module_name_list', 'acc_list', 'loss_list', 'reward_list', 'val_acc_list', 'val_loss_list', 'val_reward_list', 'param_grad_var_list', 'continuous_vg_count', 'continuous_eg_count', 'continuous_dr_count', 'at_symptom', 'dd_symptom', 'wd_symptom'])
        # metric_y = np.log10(d["val_loss"]) if type(d["val_loss"]) is float else 10  # val_loss okkk
        # metric_y = np.log10(d["param_grad_zero_rate"])  # np.log10(d["param_grad_zero_rate"]) 0.07
        # metric_y = 1 if d["cmp_sc_metric"] is not None else 0  # np.log10(d["param_grad_zero_rate"]) # cmp_sc_metric
        metric_y = get_dr_metric(d["param_grad_zero_rate"]) # okkk
        # metric_y = get_sc_metric(d["acc_list"])  # okkk
        # metric_y = get_ol_metric(d["acc_list"]) # eee okk
        # metric_y = get_veg_metric(d["param_grad_abs_ave_list"], d["module_name_flow_2dlist"], d["module_name_list"])
        id_x_dict[param_id] = metric_x
        id_y_dict[param_id] = metric_y

    print(len(id_x_dict), len(id_y_dict))
    print(max(id_x_dict.values()))
    plt.scatter(id_x_dict.values(), id_y_dict.values(),edgecolors='r')
    plt.show()


if __name__ == '__main__':
    # plot()
    # bar1()
    # bar2()
    # pie()
    # bar3()
    # plot_reproduce()
    plot_feature()
