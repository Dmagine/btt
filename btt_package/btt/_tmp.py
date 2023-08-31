import inspect
import multiprocessing
import time

from _tmpp import A


def func(s="eee"):
    while True:
        print("{}:{}".format(s, time.time()))
        time.sleep(1)


def main():
    print("Main process starts")
    names = ["aaa", "bbb", "ccc", "ddd"]
    p_list = []
    for i in range(len(names)):
        p = multiprocessing.Process(target=func, args=(names[i],))
        p.daemon = False
        p.start()
        p_list.append(p)
        print(p.pid)
    print("Main process continues")
    time.sleep(20)
    for p in p_list:
        p.terminate()
        p.join()
    print("Main process finished")


class AlgorithmConfig:
    # for: monitor_rule tuner_hpo assessor_indicator
    # btt的monitor/tuner/assessor其实都只有一个 只是具体可能存在多个(也不一定同时用)的rule/hpo/indicator！
    def __init__(self, init_args=None, class_name=None, module_path=None):
        self.name = None
        self.module_path = module_path
        self.class_name = class_name
        self.init_args = init_args

        self.class_ctor = None

    def canonicalize(self, name):
        self.name = name  ####
        self.init_args = {} if self.init_args is None else self.init_args
        if inspect.isclass(self.class_name):
            module = inspect.getmodule(self.class_name)
            self.class_ctor = self.class_name
            self.module_path = module.__file__
            self.class_name = self.class_ctor.__name__


def tmp_class():
    conf = AlgorithmConfig(init_args={"a": 1, "b": 2}, class_name=A)
    conf.canonicalize("test")
    print(conf.name, conf.module_path, conf.class_name, conf.init_args)


def tmp_dict():
    import threading

    class ThreadSafeDict(dict):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._lock = threading.Lock()

        def __len__(self):
            with self._lock:
                return super().__len__()

        def __getitem__(self, key):
            with self._lock:
                return super().__getitem__(key)

        def __setitem__(self, key, value):
            with self._lock:
                super().__setitem__(key, value)

    # 使用示例
    safe_dict = ThreadSafeDict(a=1, b=2)

    def modify_dict():
        for _ in range(10000):
            safe_dict['a'] += 1
            safe_dict['b'] -= 1

    threads = []
    for _ in range(10):
        thread = threading.Thread(target=modify_dict)
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    print(safe_dict)  # Output: {'a': 10001, 'b': -9999}


def temp__():
    class A:
        def __init__(self):
            self.a = 1

        def __hello(self):
            print("value is {}".format(self.a))

    a = A()
    a.__hello()


def my_func():
    print("hello")


def tmp_func_ctor():
    def my_func2():
        print("hello")
    class TrialFuncConfig:
        def __init__(self, func_ctor):
            self.name = func_ctor.__name__
            self.path = inspect.getmodule(func_ctor).__file__
            self.ctor = func_ctor

    a = TrialFuncConfig(func_ctor=my_func2)
    print(a.name, a.path, a.ctor)


if __name__ == '__main__':
    tmp_func_ctor()
