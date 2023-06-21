# 提取txt文件中的所有浮点数字

import os

def latex():
    file_name = 'tmp.txt'
    with open(file_name, 'r') as f:
        # 每行提取浮点数
        for line in f.readlines():
            pre_flag = False
            for i in range(len(line)):
                c = line[i]
                if c in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.']:
                    pre_flag = True
                    print(c, end='')
                else:
                    if pre_flag:
                        print('\t', end='')
                        pre_flag = False
                pass
            print()
def calc_rate():
    file_name = 'tmp.txt'
    with open(file_name, 'r') as f:
        # 两行数 计算下一行比上一行增加的比率
        pre_line = None
        for line in f.readlines():
            if pre_line is not None:
                pre_nums = [float(x) for x in pre_line.split()]
                nums = [float(x) for x in line.split()]
                for i in range(len(nums)):
                    print('%.1f' % ((nums[i] - pre_nums[i]) / pre_nums[i] *100), end='\t')
                print()
            pre_line = line

if __name__ == "__main__":
    calc_rate()
