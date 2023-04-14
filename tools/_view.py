import os
from datetime import datetime
import time


def view_lastn_list(lst=None):
    dir_path = "../script_log/"
    tmp_file_path = "_view_tmp.txt"

    lst = [x for x in range(1,3+1)]
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

    # os.system("nnictl stop --all ")
    count = 0
    for i in range(len(file_paths)):
        os.system("nnictl stop " + " --port " + str(8081 + int(i)))
    for i in range(len(file_paths)):
        f = open(dir_path + file_paths[i])
        line = f.readline()
        f.close()
        exp_id = line[-13:-5]
        # _idx = file_paths[i][file_paths[i].rfind("_") + 1:-4]
        print(file_paths[i], exp_id, i)
        os.system("nnictl stop " + exp_id)
        os.system("nnictl view " + exp_id + " --port " + str(8081 + int(i)))
        count += 1
    os.system("nnictl stop --port " + str(8081 + int(len(file_paths))))

if __name__ == '__main__':
    view_lastn_list()
