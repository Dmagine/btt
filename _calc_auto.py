import os
import sqlite3
from datetime import datetime
import time

import yaml


def calc_auto(lst=None):
    dir_path = "./script_log/"
    tmp_file_path = "_calc_auto_tmp.txt"

    lst = [x + 4 for x in [0, 1, 2, 3, 4, 5]]
    lst.reverse()
    os.system("echo > " + tmp_file_path)
    for last_num in lst:
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

    # id_lst = ["zd74o31m",
    #           "1x6u8hle",
    #           "6qt53xbl",
    #           "8lwcnju5",
    #           "bhoaeudf",
    #           "bh85n471",
    #           "k73r1fum",
    #           "15t3ipn2",
    #           "l4n6md1j",
    #           "ub3z7ywc"] #anneal

    s = ""
    os.system("echo > " + tmp_file_path)
    for i in range(len(id_lst)):
        r = calc_top(id_lst[i])
        print(id_lst[i], r)
        line_lst = [id_lst[i]]
        line_lst.extend(r)
        line_str = get_line(line_lst)
        s += line_str + "\n"
    f = open(tmp_file_path, "w")
    f.write(s)
    f.close()


def calc_top(exp_id):
    db_path = os.path.join("~/nni-experiments", exp_id, "db/nni.sqlite")
    # print(db_path)

    code_path = "/home/peizhongyi/Pycharm-Projects-cenn/nni_at_assessor"
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
    lst = lst[0:top_n]
    return lst


def get_line(lst):
    s = ""
    for i in range(len(lst)):
        s += '\t' * i + str(lst[i]) + "\n"
    return s


if __name__ == '__main__':
    calc_auto()
