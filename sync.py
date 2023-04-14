import os


def main():
    prefix = "192.168.6."
    suffix = ["106", "105", "107", "109", "113"]
    source = "/Users/admin/PycharmProjects/NNI-test/*"
    dest = "~/Pycharm-Projects-cenn/nni_at_assessor"
    for i in range(len(suffix)):
        user = "cenzhiyao@" if suffix != "106" else "peizhongyi@"
        tmp = user + prefix + suffix[i] + ":" + dest
        os.system("scp -r " + source + " " + tmp)

if __name__ == '__main__':
    main()
