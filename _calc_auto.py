import os
from datetime import datetime
import time


def calc_auto(lst=None):
    dir_path = "./script_log/"
    tmp_file_path = "_calc_auto_ids.txt"

    lst = [1,2,3,4,5]
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
    count = 0
    for i in range(len(file_paths)):
        f = open(dir_path + file_paths[i])
        line = f.readline()
        f.close()
        exp_id = line[-13:-5]
        id_lst.append(exp_id)
    print(id_lst)


if __name__ == '__main__':
    calc_auto()

