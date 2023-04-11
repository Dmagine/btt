import os.path
import sqlite3

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.stats import gaussian_kde, stats


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
                        b = (np.log10(beta3) - mid) / (2 ** level)
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
        # metric_y = get_dr_metric(d["param_grad_zero_rate"]) # okkk
        metric_y = d["param_grad_zero_rate"]  #
        # metric_y = get_sc_metric(d["acc_list"])  # okkk
        # metric_y = get_ol_metric(d["acc_list"]) # eee okk
        # metric_y = get_veg_metric(d["param_grad_abs_ave_list"], d["module_name_flow_2dlist"], d["module_name_list"])
        id_x_dict[param_id] = metric_x
        id_y_dict[param_id] = metric_y

    print(len(id_x_dict), len(id_y_dict))
    print(max(id_x_dict.values()))
    plt.scatter(id_x_dict.values(), id_y_dict.values(), edgecolors='r')
    plt.show()


def _get_partial_ave(param_grad_abs_ave_list, i_lst):
    v_lst = param_grad_abs_ave_list
    tmp_lst = []
    for i in range(len(v_lst)):
        if i in i_lst:
            tmp_lst.append(v_lst[i])
    return sum(tmp_lst) / len(tmp_lst)


def get_quotient_list(param_grad_abs_ave_list, module_name_list, module_name_flow_2dlist):
    for i in range(len(param_grad_abs_ave_list)):
        if type(param_grad_abs_ave_list[i]) is str:
            param_grad_abs_ave_list[i] = np.inf
        param_grad_abs_ave_list[i] = np.array(param_grad_abs_ave_list[i])
    lst = []
    for module_name_flow_list in module_name_flow_2dlist:
        module_idx_list_flow_list = []
        for name in module_name_flow_list:
            idx_lst = []
            for idx in range(len(module_name_list)):
                full_name = module_name_list[idx]
                if name in full_name and full_name.index(name) == 0:  # prefix
                    idx_lst.append(idx)
            module_idx_list_flow_list.append(idx_lst)
        for i in range(len(module_idx_list_flow_list) - 1):
            idx_1st1 = module_idx_list_flow_list[i]
            idx_lst2 = module_idx_list_flow_list[i + 1]
            v1 = _get_partial_ave(param_grad_abs_ave_list, idx_1st1)
            v2 = _get_partial_ave(param_grad_abs_ave_list, idx_lst2)
            lst.append(v1 / v2)
    return lst


def get_ol_list(acc_list):
    maximum_list = []
    minimum_list = []
    lst = []
    for i in range(len(acc_list)):
        if i == 0 or i == len(acc_list) - 1:
            continue
        if acc_list[i] - acc_list[i - 1] >= 0 and acc_list[i] - acc_list[i + 1] >= 0:
            maximum_list.append(acc_list[i])
        if acc_list[i] - acc_list[i - 1] <= 0 and acc_list[i] - acc_list[i + 1] <= 0:
            minimum_list.append(acc_list[i])
    for i in range(min(len(maximum_list), len(minimum_list))):
        lst.append(maximum_list[i] - minimum_list[i])
    return lst


def clean_list(lst):
    for i in range(len(lst)):
        lst[i] = clean_val(lst[i])
    return lst


def clean_val(s):
    if s == 'inf':
        return np.inf
    elif s == '-inf':
        return -np.inf
    elif s == 'NaN' or type(s) is str:
        return np.nan
    else:
        return float(s)


def clean_dict(id_x_dict, id_y_dict):
    nan_k_list = []
    inf_k_list = []
    minus_inf_k_list = []
    valid_v_list = [v for v in id_y_dict.values() if v != np.inf and v != -np.inf and v != np.nan]
    for k, v in id_y_dict.items():
        if v == -np.inf:
            minus_inf_k_list.append(k)
            print("-inf")
        if v == np.inf:
            inf_k_list.append(k)
            print("inf")
        if np.isnan(v):
            nan_k_list.append(k)
            print("nan")
    gap = (max(valid_v_list) - min(valid_v_list)) * 0.1  ######
    for k in id_y_dict.keys():
        if k in minus_inf_k_list:
            id_y_dict[k] = min(valid_v_list) - gap
        if k in inf_k_list:
            id_y_dict[k] = max(valid_v_list) + gap

    id_x_dict = {k: v for k, v in id_x_dict.items() if k not in nan_k_list}
    id_y_dict = {k: v for k, v in id_y_dict.items() if k not in nan_k_list}
    return id_x_dict, id_y_dict


def plot_testbed():
    scene_name_list = ["test",
                       "test"]  # "test" "mnistlstm", "cifar10res18", "fashionlenet5","cifar100vgg16", "catstrans"
    scene_id_list = ["ihaxrjm7", "d8kxgme9"]  # "ihaxrjm7" "tpk1j376", "m61of0s7", "h3p8wkex", "1e84bp09", "vfisu80c"

    for scene_idx in range(len(scene_name_list)):
        print(scene_name_list[scene_idx])
        desk_dir = "/Users/admin/Desktop/"
        db_path = os.path.join(desk_dir, scene_name_list[scene_idx], scene_id_list[scene_idx])
        db_pathh = os.path.join("./", scene_name_list[scene_idx])
        os.system(" ".join(["cp", db_path, db_pathh]))
        conn = sqlite3.connect(db_pathh)
        cur = conn.cursor()

        # sql = "SELECT * FROM TrialJobEvent WHERE event='WAITING'"
        # cur.execute(sql)
        # values = cur.fetchall()
        # relu_id_list = []
        # for i in range(len(values)):  # final ??? seq include 0
        #     trial_id = values[i][1]
        #     d = yaml.load(values[i][3], Loader=yaml.FullLoader)
        #     print(d["parameters"])
        #     use_relu = d["parameters"]["act_func"] in ["relu", "relu6"]
        #     if use_relu:
        #         relu_id_list.append(trial_id)
        # # print(relu_id_list)

        sql = "SELECT * FROM MetricData WHERE type='FINAL'"
        cur.execute(sql)
        print("begin to fetch data x:")
        values = cur.fetchall()
        id_x_dict = {}  # default
        for i in range(len(values)):  # final ??? seq include 0
            trial_id = values[i][1]
            # if trial_id in relu_id_list:
            #     continue
            d = yaml.load(eval(values[i][5]), Loader=yaml.FullLoader)
            metric_x = d["default"]
            id_x_dict[trial_id] = metric_x
        top_x_val = np.percentile(list(id_x_dict.values()), 99)  # 从小到大排 第95% val_acc
        good_x_val = np.percentile(list(id_x_dict.values()), 95)
        care_x_val = np.percentile(list(id_x_dict.values()), 90)

        sql = "SELECT * FROM MetricData WHERE type='PERIODICAL' and sequence=19"  ###### and sequence=19
        # WHERE type='FINAL' 'PERIODICAL' limit 10000
        cur.execute(sql)
        print("begin to fetch data y:")
        values = cur.fetchall()
        id_y_dict = {}
        print("begin to processing data:")
        # nan_count = 0
        for i in range(len(values)):  # final ??? seq include 0
            trial_id = values[i][1]
            d = yaml.load(eval(values[i][5]), Loader=yaml.FullLoader)
            # metric_y = d["param_grad_zero_rate"] # d["param_grad_0rate"]
            # metric_y = d["acc"]
            # metric_y = id_x_dict[param_id] - d["default"]
            # metric_y = np.log10(d["loss"]) if type(d["loss"]) is not str else np.inf
            # lst = d["param_grad_abs_ave_list"]
            # val = sum(lst) / len(lst) if type(lst[0]) is not str else np.inf
            # metric_y = np.log10(val)
            # param_grad_abs_ave_list = d["param_grad_abs_ave_list"]
            # module_name_list = d["module_name_list"]
            # module_name_flow_2dlist = d["module_name_flow_2dlist"]
            # lst = get_quotient_list(param_grad_abs_ave_list, module_name_list, module_name_flow_2dlist)
            # val = sum(lst)/len(lst)
            # metric_y = np.log10(val)

            # 新指标
            # metric_y = d["param_grad_0rate"]  # (0,1)
            # metric_y = np.log10(clean_val(d["param_grad_abs_ave"]))  # ok

            # metric_y = clean_val(d["param_grad_ave"])  # ok hard to scale half...
            # metric_y = np.log10(clean_val(d["param_grad_var"]))  # ok (-80,20)
            # metric_y = clean_val(d["param_grad_skew"]) # (-1,1) half...
            # metric_y = clean_val(d["param_grad_kurt"])  # (0,50) no!!!
            # metric_y = clean_val(d["param_ave"]) # (-10,10) no!!!
            # metric_y = np.log10(clean_val(d["param_var"])) # (-80,20) ok
            # metric_y = clean_val(d["param_skew"]) # (-1,1) no!!!
            # metric_y = clean_val(d["param_kurt"])  #  (0,50) no!!!
            id_y_dict[trial_id] = metric_y
        y_range = (0, 1)
        valid_v_list = [v for v in id_y_dict.values() if v != np.inf and v != -np.inf and v != np.nan]
        for k, v in id_y_dict.items():
            if v == -np.inf:
                id_y_dict[k] = min(valid_v_list)
                print("-inf")
            if v == np.inf:
                id_y_dict[k] = max(valid_v_list)
                print("inf")
            if np.isnan(v):
                # id_y_dict[k] = max(valid_v_list)
                id_x_dict.pop(k)
                print("nan")
        print("begin to split data:")
        y_top, y_good, y_all, y_care = [], [], [], []
        for key, value in id_x_dict.items():
            if value >= top_x_val:
                y_top.append(id_y_dict[key])
            if value >= good_x_val:
                y_good.append(id_y_dict[key])
            if value >= care_x_val:
                y_care.append(id_y_dict[key])
            y_all.append(id_y_dict[key])
        print("begin to plot data:")
        plt.subplot(1, len(scene_name_list), scene_idx + 1)
        # y_range = (min(id_y_dict.values()), max(id_y_dict.values()))
        # (0,1) (-3,2) (-10, 5) (0,0.5) (-0.25,0.25) (-1,1) (0,50) (-50,20)
        y_bins = 30  #
        plt.hist(y_all, range=y_range, bins=y_bins, color='grey', alpha=0.8)
        plt.hist(y_care, range=y_range, bins=y_bins, color='b', alpha=0.8)
        plt.hist(y_good, range=y_range, bins=y_bins, color='g', alpha=0.8)
        plt.hist(y_top, range=y_range, bins=y_bins, color='r', alpha=0.8)
    plt.show()


def print_distribution(y_list):
    print(" ".join(["min:", str(min(y_list)), "lower:", str(np.percentile(y_list, 25)),
                    "mean:", str(np.mean(y_list)), "median:", str(np.median(y_list)),
                    "upper:", str(np.percentile(y_list, 75)), "max:", str(max(y_list))]))


def calc_skew(lst):
    return stats.skew(lst)


def calc_kurt(lst):
    return stats.kurtosis(lst)


def get_adjacent_quotient_lst(lst):
    return [lst[i] / lst[i + 1] for i in range(len(lst) - 1)]


def plot_merge_epoch():
    scene_name_list = ["fashionlenet5", "cifar10res18"]
    scene_id_list = ["ihaxrjm7", "v542mpos"]
    # "test" "mnistlstm", "cifar10res18", "fashionlenet5","cifar100vgg16", "catstrans"
    # "ihaxrjm7" "tpk1j376", "m61of0s7", "h3p8wkex", "1e84bp09", "vfisu80c"
    max_seq = 0

    for scene_idx in range(len(scene_name_list)):
        print(scene_name_list[scene_idx])
        desk_dir = "/Users/admin/Desktop/new/"
        db_path = os.path.join(desk_dir, scene_name_list[scene_idx], scene_id_list[scene_idx])
        db_pathh = os.path.join("./", scene_name_list[scene_idx])
        os.system(" ".join(["cp", db_path, db_pathh]))
        conn = sqlite3.connect(db_pathh)
        cur = conn.cursor()
        sql = "SELECT * FROM MetricData WHERE type='FINAL'"
        cur.execute(sql)
        print("begin to fetch data x:")
        values = cur.fetchall()
        id_x_dict = {}  # default
        id_y_dict = {}
        for i in range(len(values)):  # final ??? seq include 0
            trial_id = values[i][1]
            d = yaml.load(eval(values[i][5]), Loader=yaml.FullLoader)
            metric_x = clean_val(d["default"])  #####
            id_x_dict[trial_id] = metric_x
            id_y_dict[trial_id] = [0] * (max_seq + 1)  # init

        sql = "SELECT * FROM MetricData WHERE type='PERIODICAL' and sequence<=" + str(max_seq)  # 模拟小数据集 0 1 2
        # and sequence=19 'FINAL' 'PERIODICAL' limit 10000
        cur.execute(sql)
        print("begin to fetch data y:")
        values = cur.fetchall()
        print("begin to processing data:")
        for i in range(len(values)):  # final ??? seq include 0
            trial_id = values[i][1]
            seq = values[i][4]
            d = yaml.load(eval(values[i][5]), Loader=yaml.FullLoader)

            # metric_y = clean_val(d["param_grad_0rate"])
            # metric_y = np.log10(clean_val(d["param_grad_abs_ave"]))

            # 感觉这个是最好的：
            # EG: skew负很多/kurt很大/max&min&median都大 "偏科学习"
            # VG: skew正很多/kurt很大/max&min&median都小 "偏科学习"
            # DR&SC：max&min&median都很小/skew绝对值小/kurt绝对值小 "不学习"
            # OL：max&min&median都很大 "乱学习"
            # 正常：max&min&median不大不小/skew绝对值小/kurt绝对值小 "正常学习"
            # metric_y = calc_skew(clean_list(d["param_grad_abs_ave_list"])) # 一般
            # metric_y = calc_kurt(clean_list(d["param_grad_abs_ave_list"]))# 一般
            metric_y = np.median(clean_list(d["param_grad_abs_ave_list"]))  # 一般
            # metric_y = np.max(clean_list(d["param_grad_abs_ave_list"]))# 一般
            # metric_y = np.min(clean_list(d["param_grad_abs_ave_list"]))# 一般

            # lst = get_adjacent_quotient_lst(clean_list(d["param_grad_abs_ave_list"]))
            # metric_y = calc_skew(lst)
            # metric_y = calc_kurt(lst)
            # metric_y = np.median(lst)
            # metric_y = np.max(lst)
            # metric_y = np.min(lst)

            # metric_y = clean_val(d["param_grad_ave"])  # ... 跨场景难以scale
            # metric_y = np.log10(clean_val(d["param_grad_var"]))
            # metric_y = clean_val(d["param_grad_skew"])
            # metric_y = np.log10(clean_val(d["param_grad_kurt"]))
            # metric_y = clean_val(d["param_ave"])
            # metric_y = np.log10(clean_val(d["param_var"]))
            # metric_y = clean_val(d["param_skew"])
            # metric_y = clean_val(d["param_kurt"])

            # metric_y = clean_val(d["loss"])
            # metric_y = clean_val(d["acc"])
            # metric_y = clean_val(d["val_loss"])
            # metric_y = clean_val(d["val_acc"])
            id_y_dict[trial_id][seq] = metric_y
        id_y_dict = {k: np.mean(v) for k, v in id_y_dict.items()}
        id_x_dict, id_y_dict = clean_dict(id_x_dict, id_y_dict)
        print("x len:", len(id_x_dict))
        x_val_99 = np.percentile(list(id_x_dict.values()), 99)  # 从小到大排 第95% val_acc
        x_val_95 = np.percentile(list(id_x_dict.values()), 95)
        x_val_90 = np.percentile(list(id_x_dict.values()), 90)
        x_val_50 = np.percentile(list(id_x_dict.values()), 50)
        x_val_25 = np.percentile(list(id_x_dict.values()), 25)
        x_val_10 = np.percentile(list(id_x_dict.values()), 10)
        y_99_up = [id_y_dict[key] for key, x in id_x_dict.items() if x >= x_val_99]
        y_95_up = [id_y_dict[key] for key, x in id_x_dict.items() if x >= x_val_95]
        y_90_up = [id_y_dict[key] for key, x in id_x_dict.items() if x >= x_val_90]

        y_50_down = [id_y_dict[key] for key, x in id_x_dict.items() if x < x_val_50]
        y_25_down = [id_y_dict[key] for key, x in id_x_dict.items() if x < x_val_25]
        y_10_down = [id_y_dict[key] for key, x in id_x_dict.items() if x < x_val_10]
        y_0 = list(id_y_dict.values())
        y_upper_bound, y_lower_bound = np.percentile(y_0, 95), np.percentile(y_0, 5)
        gap = (y_upper_bound - y_lower_bound) * 0.1
        y_upper_bound, y_lower_bound = y_upper_bound + gap, y_lower_bound - gap
        # print("y_upper_bound:", y_upper_bound, "y_lower_bound:", y_lower_bound)
        print("begin to plot data:")
        plt.subplot(1, len(scene_name_list), scene_idx + 1)
        y_bins = 20  #
        y_range = (y_lower_bound, y_upper_bound)
        x = np.linspace(y_lower_bound, y_upper_bound, 100)
        # plt.hist(y_100, range=y_range, bins=y_bins, color='grey', alpha=0.8)
        # plt.hist(y_90, range=y_range, bins=y_bins, color='b', alpha=0.8)
        # plt.hist(y_95, range=y_range, bins=y_bins, color='g', alpha=0.8)
        # plt.hist(y_99, range=y_range, bins=y_bins, color='r', alpha=0.8)
        # plt.hist(y_0, range=y_range, bins=y_bins, color='grey', density=True, alpha=0.5)
        # plt.plot(x, gaussian_kde(y_0, bw_method=None)(x), color='grey', linewidth=1)
        # print("begin to print distribution:")
        # print_distribution(y_0)
        # target bad
        plt.hist(y_50_down, range=y_range, bins=y_bins, color='b', density=True, alpha=0.5)
        plt.plot(x, gaussian_kde(y_50_down, bw_method=None)(x), color='b', linewidth=1)
        print("begin to print distribution:")
        print_distribution(y_50_down)
        # good
        plt.hist(y_95_up, range=y_range, bins=y_bins, color='r', density=True, alpha=0.5)
        plt.plot(x, gaussian_kde(y_95_up, bw_method=None)(x), color='r', linewidth=1)
        print("begin to print distribution:")
        print_distribution(y_95_up)

    plt.show()


def plot_last_layer():
    scene_name_list = ["fashionlenet5", "cifar10res18"]
    scene_id_list = ["ihaxrjm7", "v542mpos"]

    for scene_idx in range(len(scene_name_list)):
        print(scene_name_list[scene_idx])
        desk_dir = "/Users/admin/Desktop/new/"
        db_path = os.path.join(desk_dir, scene_name_list[scene_idx], scene_id_list[scene_idx])
        db_pathh = os.path.join("./", scene_name_list[scene_idx])
        os.system(" ".join(["cp", db_path, db_pathh]))
        conn = sqlite3.connect(db_pathh)
        cur = conn.cursor()
        sql = "SELECT * FROM MetricData WHERE type='FINAL'"
        cur.execute(sql)
        print("begin to fetch data x:")
        values = cur.fetchall()
        id_x_dict = {}  # default
        id_y_dict = {}
        for i in range(len(values)):  # final ??? seq include 0
            trial_id = values[i][1]
            d = yaml.load(eval(values[i][5]), Loader=yaml.FullLoader)
            metric_x = clean_val(d["default"])  #####
            id_x_dict[trial_id] = metric_x
            id_y_dict[trial_id] = [0]  # init

        sql = "SELECT * FROM MetricData WHERE type='PERIODICAL' and sequence=0 "  # 模拟小数据集 0 1 2
        cur.execute(sql)
        print("begin to fetch data y:")
        values = cur.fetchall()
        print("begin to processing data:")
        for i in range(len(values)):  # final ??? seq include 0
            trial_id = values[i][1]
            seq = values[i][4]
            d = yaml.load(eval(values[i][5]), Loader=yaml.FullLoader)
            # metric_y = clean_val(d["param_grad_abs_ave_list"][-1])  # y_lower_bound, y_upper_bound = 0, 0.05
            # metric_y = clean_val(d["param_grad_ave_list"][-1])
            # metric_y = clean_val(d["param_grad_var_list"][-1]) # y_lower_bound, y_upper_bound = 0, 0.01
            # metric_y = clean_val(d["param_grad_skew_list"][-1])
            # metric_y = clean_val(d["param_grad_kurt_list"][-1])
            # metric_y = clean_val(d["param_ave_list"][-1])
            # metric_y = clean_val(d["param_var_list"][-1])
            # metric_y = clean_val(d["param_skew_list"][-1])
            metric_y = clean_val(d["param_kurt_list"][-1])

            id_y_dict[trial_id][seq] = metric_y
        id_y_dict = {k: np.mean(v) for k, v in id_y_dict.items()}
        id_x_dict, id_y_dict = clean_dict(id_x_dict, id_y_dict)
        print("x len:", len(id_x_dict))
        x_val_99 = np.percentile(list(id_x_dict.values()), 99)  # 从小到大排 第95% val_acc
        x_val_95 = np.percentile(list(id_x_dict.values()), 95)
        x_val_90 = np.percentile(list(id_x_dict.values()), 90)
        x_val_50 = np.percentile(list(id_x_dict.values()), 50)
        x_val_25 = np.percentile(list(id_x_dict.values()), 25)
        x_val_10 = np.percentile(list(id_x_dict.values()), 10)
        y_99_up = [id_y_dict[key] for key, x in id_x_dict.items() if x >= x_val_99]
        y_95_up = [id_y_dict[key] for key, x in id_x_dict.items() if x >= x_val_95]
        y_90_up = [id_y_dict[key] for key, x in id_x_dict.items() if x >= x_val_90]

        y_50_down = [id_y_dict[key] for key, x in id_x_dict.items() if x < x_val_50]
        y_25_down = [id_y_dict[key] for key, x in id_x_dict.items() if x < x_val_25]
        y_10_down = [id_y_dict[key] for key, x in id_x_dict.items() if x < x_val_10]

        y_0 = list(id_y_dict.values())
        y_upper_bound, y_lower_bound = np.percentile(y_0, 95), np.percentile(y_0, 5)
        gap = (y_upper_bound - y_lower_bound) * 0.1
        y_upper_bound, y_lower_bound = y_upper_bound + gap, y_lower_bound - gap
        print("begin to plot data:")
        plt.subplot(1, len(scene_name_list), scene_idx + 1)
        y_bins = 20  #
        # y_lower_bound, y_upper_bound = 0, 0.05
        y_range = (y_lower_bound, y_upper_bound)
        x = np.linspace(y_lower_bound, y_upper_bound, 100)

        # target bad
        target = y_50_down
        plt.hist(target, range=y_range, bins=y_bins, color='b', density=True, alpha=0.5)
        plt.plot(x, gaussian_kde(target, bw_method=None)(x), color='b', linewidth=1)
        print("begin to print distribution:")
        print_distribution(target)
        # good
        target = y_95_up
        plt.hist(target, range=y_range, bins=y_bins, color='r', density=True, alpha=0.5)
        plt.plot(x, gaussian_kde(target, bw_method=None)(x), color='r', linewidth=1)
        print("begin to print distribution:")
        print_distribution(target)
    plt.show()


def plot_benchmark():
    import scipy.stats as stats
    scene_name_list = ["mnistlenet_large"]
    scene_id_list = ["ou7xnzlm"]
    max_seq = 0
    for scene_idx in range(len(scene_name_list)):
        print(scene_name_list[scene_idx])
        desk_dir = "/Users/admin/Desktop/deep-bo-benchmark/"
        db_path = os.path.join(desk_dir, scene_name_list[scene_idx], scene_id_list[scene_idx])
        db_pathh = os.path.join("./", scene_name_list[scene_idx])
        os.system(" ".join(["cp", db_path, db_pathh]))
        conn = sqlite3.connect(db_pathh)
        cur = conn.cursor()
        sql = "SELECT * FROM MetricData WHERE type='FINAL'"
        cur.execute(sql)
        print("begin to fetch data x:")
        values = cur.fetchall()
        id_x_dict = {}  # default
        id_y_list_dict = {}
        for i in range(len(values)):  # final ??? seq include 0
            trial_id = values[i][1]
            d = yaml.load(eval(values[i][5]), Loader=yaml.FullLoader)
            id_x_dict[trial_id] = float(d)
            id_y_list_dict[trial_id] = [0] * 15

        sql = "SELECT * FROM MetricData WHERE type='PERIODICAL' "  # 模拟小数据集 0 1 2
        # and sequence=19 'FINAL' 'PERIODICAL' limit 10000
        cur.execute(sql)
        print("begin to fetch data y:")
        values = cur.fetchall()
        print("begin to processing data:")
        for i in range(len(values)):  # final ??? seq include 0
            trial_id = values[i][1]
            seq = values[i][4]
            d = yaml.load(eval(values[i][5]), Loader=yaml.FullLoader)
            if trial_id not in id_y_list_dict:
                continue
            id_y_list_dict[trial_id][seq] = float(d)

        print("begin to plot data:")
        p_list = [99, 95, 90, 50, 0]
        c_list = ['r', 'g', 'b', 'y', 'k']
        for i in range(len(p_list)):
            x_val = np.percentile(list(id_x_dict.values()), p_list[i])
            y_matrix = np.array([id_y_list_dict[key] for key, x in id_x_dict.items() if x >= x_val])
            x_list = np.array([id_x_dict[key] for key, x in id_x_dict.items() if x >= x_val])
            plot_y_list = [stats.spearmanr(x_list,y_matrix[:,epoch_i])[0] for epoch_i in range(15)]
            plot_x_list = np.arange(0, 15, 1)
            plt.plot(plot_x_list, plot_y_list, color=c_list[i], label=str(p_list[i]) + "th-up")
            plt.legend()
        plt.show()
        x_list = list(id_x_dict.values())
        x_list.sort()
        x_list.reverse()
        plot_x_list = np.linspace(0, 1, len(x_list))
        plot_y_list = x_list
        plt.plot(plot_x_list, plot_y_list, color='b', label="x=y")
        plt.show()


if __name__ == '__main__':
    # plot()
    # bar1()
    # bar2()
    # pie()
    # bar3()
    # plot_reproduce()
    # plot_feature()
    # plot_testbed()
    # plot_merge_epoch()
    # plot_last_layer()
    plot_benchmark()
