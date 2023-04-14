import os

import yaml


def calc():
    import numpy as np

    tmpp_file_path = "_calc_tmp.txt"
    f = open(tmpp_file_path)
    s = f.read().strip()
    f.close()

    num_list = [float(x.strip()) for x in s.split("\n")]
    print(num_list)
    print(len(num_list))

    tmppp_file_path = "_calc_tmp.txt"
    os.system("echo >" + tmppp_file_path)
    s_list = []
    for i in range(3):
        l = []
        for idx in range(len(num_list)):
            if idx % 3 == i:
                l.append(num_list[idx])
        # s = "%.4f±%.4f" % (float(np.mean(l)), float(np.var(l)))
        s = "%.4f" % (float(np.mean(l)))
        s_list.append(s)
        print(s)
    s = "%d" % (float(np.mean(num_list)))
    print(s)

    ss = ""
    for i in range(len(s_list)):
        ss += "\t" * i + s_list[i] + "\n"
    f = open(tmppp_file_path, "w")
    f.write(ss)
    f.close()


if __name__ == '__main__':
    calc()
