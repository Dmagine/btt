import os
import sqlite3
from datetime import datetime
import time

import numpy as np
import yaml


def calc_auto(acc_lst=None):
    dir_path = "../script_log/"
    tmp_file_path = "_calc_auto_tmp.txt"

    acc_lst = [x for x in range(4)]
    acc_lst.reverse()
    os.system("echo > " + tmp_file_path)
    for last_num in acc_lst:
        os.system("ls " + dir_path + " -lrt | tail -n " + str(last_num) +
                  " | head -n 1 | awk '{print $9}'| xargs echo >> " + tmp_file_path)

    f = open(tmp_file_path)
    s = f.read().strip()
    f.close()
    file_paths = [x.strip() for x in s.split("\n")]
    print(file_paths)

    id_lst = []
    for i in range(len(file_paths)):
        f = open(dir_path + file_paths[i])
        line = f.readline()
        f.close()
        exp_id = line[-13:-5]
        id_lst.append(exp_id)
    print(id_lst)

    # id_lst = ["z1v4r0fa",
    #           "0pktdxql",
    #           "kmw9lnxy",
    #           "85woxflt",
    #           "r9uledv1",
    #           "z7lqc6fk",
    #           "jpgacixk",
    #           "xm3ebp8i",
    #           "i7dj5fks",
    #           "nb3hlfyx"] #tpe

    # id_lst = ["gfnwz4vu","3ta1cw56","6iwapb78","h9rzj54y","merz6xs3","hltw50qu"]

    s = ""
    os.system("echo > " + tmp_file_path)

    s += get_line(id_lst)

    num_lst = []
    data_lst = []
    for i in range(len(id_lst)):
        acc_lst, num = calc_top_acc_num(id_lst[i])
        print(id_lst[i], acc_lst, "\t", num, "\t", file_paths[i])
        num_lst.append(num)
        data_lst.extend(acc_lst)
        # line_lst = [id_lst[i]]
        line_lst = []
        line_lst.extend(acc_lst)
        line_str = get_line(line_lst)
        s += line_str + "\n"
    s += get_line(num_lst) + "\n"
    s += get_line([round(sum(num_lst) / len(num_lst))]) + "\n"
    # print(data_lst)
    s += get_line(calc_topk_ave_var(data_lst))
    f = open(tmp_file_path, "w")
    f.write(s)
    f.close()


def calc_topk_ave_var(data_lst, k=3):
    print()
    s_list = []
    for i in range(0, 3):  # 0 1 2 3
        l = []
        for idx in range(len(data_lst)):
            if idx % k == i:
                l.append(data_lst[idx])
        # print(l,np.var(l))
        # s = "%.4f±%.4f" % (float(np.mean(l)), float(np.var(l))) # 方差小 -》 0。0000
        s = "%.4f" % (float(np.mean(l)))
        s_list.append(s)
        print(s)
    return s_list


def calc_top_acc_num(exp_id):
    db_path = os.path.join("~/nni-experiments", exp_id, "db/nni.sqlite")
    code_path = "/"
    os.system(" ".join(["cp", db_path, code_path]))
    db_path = os.path.join(code_path, "nni.sqlite")

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    sql = "SELECT * FROM MetricData WHERE type = 'FINAL'"
    cur.execute(sql)
    values = cur.fetchall()
    top_n = 3
    lst = []
    for i in range(len(values)):
        d = yaml.load(eval(values[i][5]), Loader=yaml.FullLoader)
        if type(d) is dict:
            lst.append(d["default"])
        else:
            lst.append(d)
    lst.sort()
    lst.reverse()
    num = len(lst)
    #####
    # lst = lst[0:top_n]
    lst = [lst[0], lst[4], lst[9]]

    sql = "SELECT * FROM MetricData WHERE type = 'PERIODICAL'"
    cur.execute(sql)
    values = cur.fetchall()
    s = set()
    for i in range(len(values)):
        s.add(values[i][1])
    num = len(s)

    return lst, num


def get_line(lst):
    s = ""
    for i in range(len(lst)):
        s += '\t' * i + str(lst[i]) + "\n"
    return s


if __name__ == '__main__':
    calc_auto()
