import matplotlib.pyplot as plt


# # 第一个表示x轴,第二个列表表示y轴
# plt.plot([1, 0, 9], [4, 5, 6])
# plt.show()
#
# # plt.plot(x,y)
# # 对折线进行修饰
# # color设置为红色，alpha设置为透明度，linestyle表示线的样式，linewidth表示线的宽度
# # color还可以设置为16进制的色值，可在网上查看各种颜色对应的色值
# plt.plot(x, y, color='red', alpha=0.5, linestyle='--', linewidth=1)
# plt.show()
# '''线的样式
# -	实线(solid)
# --  短线(dashed)
# -.	短点相间图
# :	虚电线(dotted)
# '''


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


def bar():
    import numpy as np
    import matplotlib.pyplot as plt
    tmpp_file_path = "_graph_in.txt"
    f = open(tmpp_file_path)
    s = f.read().strip()
    f.close()

    num_list = [float(x.strip()) for x in s.split("\n")]
    print(num_list)
    print(len(num_list))

    big = ["random", "tpe", "anneal"]
    small = ["base", "atdd"]
    total_num = len(big) * len(small)
    base_lst = []
    atdd_lst = []
    fig = plt.figure(figsize=(6, 3))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    for i in range(len(big)):
        base_lst.append(num_list[len(small) * i])
        atdd_lst.append(num_list[len(small) * i + 1])
    X = np.arange(3)
    ax.bar(X + 0.00, base_lst, color='b', width=0.3)
    ax.bar(X + 0.3, atdd_lst, color='r', width=0.3)

    plt.show()

    pass


if __name__ == '__main__':
    # plot()
    # bar()
    pie()