import inspect
import logging
import os
import threading
import time
from enum import Enum
from typing import Optional, List, Dict, Callable, Tuple

from .allocator.allocator import Allocator
from .assessor.assessor import Assessor
from .trial import Trial
from .tuner.tuner import Tuner
from .utils import get_uuid, get_package_abs_dir, ParamMode, str2second


class AlgorithmConfig:
    # for: monitor_rule tuner_hpo assessor_indicator
    # btt的monitor/tuner/assessor其实都只有一个 只是具体可能存在多个(也不一定同时用)的rule/hpo/indicator！
    def __init__(self, init_args=None, class_name=None, module_path=None):
        self.name: Optional[str] = None
        self.module_path: Optional[os.PathLike, str] = module_path
        self.class_name: Optional[str | Callable] = class_name
        self.init_args: Optional[dict] = init_args

        self.class_ctor = None

    def canonicalize(self, name):
        self.name = name  ####
        self.init_args = {} if self.init_args is None else self.init_args
        if inspect.isclass(self.class_name):
            module = inspect.getmodule(self.class_name)
            self.class_ctor = self.class_name
            self.module_path = module.__file__
            self.class_name = self.class_ctor.__name__


class TunerHpoConfig(AlgorithmConfig):
    # max_trial_number max_exp_duration
    # seed = None
    # # 优先级别：list前面的优先级高，之后exp_dur优先级高于trial_num
    # exp_dur_not_trial_num: bool = None  # True: exp_dur False: trial_num # 考虑到batchtuner
    # start_ratio: Optional[float] = None
    # end_ratio: Optional[float] = None

    # default: Optional[bool] = False # 0-1默认default
    def __init__(self, init_args=None, class_name=None, module_path=None,
                 start_ratio=None, end_ratio=None,
                 seed=None, dur_not_num=None):
        super().__init__(init_args, class_name, module_path)
        self.seed: Optional[int] = seed
        self.dur_not_num: Optional[bool] = dur_not_num  # ratio: exp_dur(default) or trial_num
        self.start_ratio: Optional[float] = start_ratio
        self.end_ratio: Optional[float] = end_ratio

    def canonicalize(self, rule_name):
        def judge_in_range(val_list, exp_min, exp_max, exp_type):
            for val in val_list:
                if type(val) is not exp_type:
                    raise ValueError("val should be " + str(exp_type))
                if val < exp_min or val > exp_max:
                    raise ValueError("val should be in range [" + str(exp_min) + "," + str(exp_max) + "]")

        super().canonicalize(rule_name)
        package_dir = get_package_abs_dir()
        self.seed = 529 if self.seed is None else self.seed
        self.dur_not_num = True if self.dur_not_num is None else self.dur_not_num
        if self.start_ratio is not None or self.end_ratio is not None:
            judge_in_range([self.start_ratio, self.end_ratio], 0, 1, float)
            self.start_ratio = 0.0 if self.start_ratio is None else self.start_ratio
            self.end_ratio = 1.0 if self.end_ratio is None else self.end_ratio
        self.module_path = os.path.join(package_dir, "tuner/tuner_hpo") \
            if self.module_path is None else self.module_path


class MonitorRuleConfig(AlgorithmConfig):
    def __init__(self, class_name=None, init_args=None,
                 interm_report=None, interm_default=None,
                 final_report=None, final_default=None,
                 initial_report=None,
                 module_path=None, ):
        super().__init__(init_args, class_name, module_path)
        self.interm_report: Optional[bool] = interm_report
        self.interm_default: Optional[bool] = interm_default
        self.final_report: Optional[bool] = final_report
        self.final_default: Optional[bool] = final_default
        self.initial_report: Optional[bool] = initial_report

    def canonicalize(self, name):
        super().canonicalize(name)
        package_dir = get_package_abs_dir()
        self.module_path = os.path.join(package_dir, "monitor/monitor_rule") \
            if self.module_path is None else self.module_path
        self.interm_default = False if self.interm_default is not True else self.interm_default
        self.interm_report = False if self.interm_report is not True else self.interm_report
        self.interm_report = True if self.interm_default is True else self.interm_report
        self.final_default = False if self.final_default is not True else self.final_default
        self.final_report = False if self.final_report is not True else self.final_report
        self.final_report = True if self.final_default is True else self.final_report
        self.initial_report = False if self.initial_report is not True else self.initial_report


class AssessorIndicatorConfig(AlgorithmConfig):
    # exp_dur
    # start_ratio: Optional[float] = None
    # end_ratio: Optional[float] = None #没必要徒增烦恼
    # start/end的ratio跟indicator自身内容是相关的 跟实验性质其实无关

    def __init__(self, init_args=None, class_name=None, module_path=None):
        super().__init__(init_args, class_name, module_path)

    def canonicalize(self, name):
        super().canonicalize(name)
        package_dir = get_package_abs_dir()
        self.module_path = os.path.join(package_dir, "assessor/assessor_indicator") \
            if self.module_path is None else self.module_path


# HP = namedtuple('hp', ['name', 'type', 'range', 'default'])


class Hyperparameter:
    def __init__(self, mode, _range, default, description=None):
        self.name = None
        self.mode: ParamMode = mode
        self.range: List[int | float] = _range
        self.default: int | float = default
        self.description: Optional[str] = description
        # self._miss = np.nan

    def canonicalize(self, name):
        self.name = name  ####
        if self.mode == ParamMode.Choice:
            if self.default not in self.range:
                raise ValueError("default should be in range")


class TrialFuncConfig:
    # 其实除了记录一下没什么用
    def __init__(self, func_ctor):
        self.name = func_ctor.__name__
        self.path = inspect.getmodule(func_ctor).__file__
        self.ctor = func_ctor

    def canonicalize(self):
        pass


class ExperimentConfig:
    # 给用户 方便填写 static -> dynamic 给resume的要复杂很多 以后再说。。。
    # exp_uuid: str = None # 用户不填

    exp_name: Optional[str] = None
    exp_description: Optional[str] = None
    trial_concurrency: int = None
    trial_gpu_number: int = None
    available_gpu_idx_list: List[int] = None
    max_trial_number: int = None
    max_exp_duration: Optional[int, str] = None  # int -> seconds
    tuner_config: Dict[str:TunerHpoConfig] = None
    monitor_config: Dict[str:MonitorRuleConfig] = None
    assessor_config: Dict[str:AssessorIndicatorConfig] = None
    hp_config: Dict[str:Hyperparameter] = None

    exps_dir = "./btt_experiments"
    config_file_path = "./exp_config.json"
    trial_func_config: TrialFuncConfig = None

    def __init__(self):
        pass

    def canonicalize(self):
        # 限制规则可能要更严格？type range default trans ...
        self.exp_name = "default_exp_name" if self.exp_name is None else self.exp_name
        self.exp_description = "default_exp_description" if self.exp_description is None else self.exp_description
        self.trial_concurrency = 4 if self.trial_concurrency is None else self.trial_concurrency
        self.trial_gpu_number = 0 if self.trial_gpu_number is None else self.trial_gpu_number  # 目前仅支持 0 1
        self.max_trial_number = 1000 if self.max_trial_number is None else self.max_trial_number
        self.max_exp_duration = 6 * 60 * 60 if self.max_exp_duration is None else self.max_exp_duration
        self.max_exp_duration = str2second(self.max_exp_duration)
        if self.tuner_config is None:
            raise ValueError("tuner_config should not be None")
        self.monitor_config = [] if self.monitor_config is None else self.monitor_config
        self.assessor_config = [] if self.assessor_config is None else self.assessor_config
        for hpo_name, hpo_config in self.tuner_config.items():
            hpo_config.canonicalize(hpo_name)
        for rule_name, rule_config in self.monitor_config.items():
            rule_config.canonicalize(rule_name)
        for indicator_name, indicator_config in self.assessor_config.items():
            indicator_config.canonicalize(indicator_name)
        self.trial_func_config.canonicalize()
        # absolite path
        self.exps_dir = os.path.abspath(self.exps_dir)
        self.config_file_path = os.path.abspath(self.config_file_path)

    def trans_to_json(self):
        pass


class TrialStatus(Enum):
    Waiting = 0
    Running = 1
    Finishing = 2  # thread over but wait 4 assessor and tuner
    Done = 2


# class SafeVariable:  # 多线程读写安全-》update还是不安全？？？非原子 哎呦喂
#     def __init__(self, init_value=None):
#         self.lock = threading.Lock()  # __ private 只有类内才能访问
#         self.__v = init_value
#
#     def get(self):
#         with self.lock:
#             return deepcopy(self.__v)  ## 依旧可能会出去被修改...?
#
#     def set(self, new_value):
#         with self.lock:
#             self.__v = new_value
#
#     def update(self, new_value):
#         with self.lock:
#             self.__v.update(new_value)  # 默认为dict
#
#
# class SharedTrialData:
#     # trial是完全按照顺序的 只是需要考虑 1exp读写 2trial读写 的并发问题
#     # exp读频率可能较高 其他读写应该频率都低
#     # 思考：每个var都不可能被exp或trial同时写入！！！！！！ 所以不怕update （暂时先这样吧 怪怪的
#     def __init__(self):
#         # 变量成员不可以直接访问！
#
#         # exp写 trial读
#         self.lock = threading.Lock()
#         self.__status = SafeVariable(TrialStatus.Waiting)
#         self.__device_str = SafeVariable(None)
#         self.__params = SafeVariable(None)  # param_name: param_value
#         self.__epoch_quota = SafeVariable(None)
#
#         # trial写 exp读
#         self.__params_used = SafeVariable(None)  # param_name: bool
#         self.__initial_result = SafeVariable(None)
#         self.__interm_result_list = SafeVariable(None)
#         self.__final_result = SafeVariable(None)
#
#     def set_status(self, new_status):
#         self.lock.acquire()
#         self.__status.set(new_status)
#         self.lock.release()
#
#     def get_status(self):
#         return self.__status.get()
#
#     def set_device_str(self, new_device_str):
#         self.__device_str.set(new_device_str)
#
#     def get_device_str(self):
#         return self.__device_str.get()
#
#     def set_params(self, new_params):
#         self.__params.set(new_params)
#         self.__params_used.set({param_name: False for param_name in new_params.keys()})
#
#     def get_params(self):
#         return self.__params.get()
#
#     def set_param_used(self, param_name):
#         d = self.__params_used.get()[param_name]
#         if param_name not in d:
#             raise ValueError("param_name should be in params_used")
#         d.update({param_name: True})
#         self.__params_used.set(d)
#
#     def get_params_used(self):
#         return self.__params_used.get()
#
#     def set_epoch_quota(self, new_epoch_quota):
#         self.__epoch_quota.set(new_epoch_quota)
#
#     def get_epoch_quota(self):
#         return self.__epoch_quota.get()
#
#     def set_initial_result(self, new_initial_result):
#         self.__initial_result.set(new_initial_result)
#
#     def get_initial_result(self):
#         return self.__initial_result.get()
#
#     def update_interm_result(self, new_interm_result):
#         l = self.__interm_result_list.get()
#         l.append(new_interm_result)
#         self.__interm_result_list.set(l)
#
#     def get_interm_result_list(self):
#         return self.__interm_result_list.get()
#
#     def set_final_result(self, new_final_result):
#         self.__final_result.set(new_final_result)
#
#     def get_final_result(self):
#         return self.__final_result.get()

class SharedTrialData:
    # 未来可以做数据库成表 -》
    # 后续操作方式： 1 sql数据表格 转 python对象 后都python 2 封装数据库读写操作
    # 定时获取 / 随时查询转换 / 如何保证数据库一整段的原子操作？？？以及stop善后处理？？？
    # 每个组件负责多列数据的读和一列数据的写？！exp-》status / tuner-》params / monitor ... / allocator ... / assessor
    def __init__(self):
        self.lock = threading.Lock()
        self.status = TrialStatus.Waiting
        self.device_str = None
        self.params = None  # param_name: param_value (tuner set)
        self.params_used = None  # param_name: bool (tuner set)
        self.initial_result = None
        self.interm_result_list = None
        self.final_result = None

        self.assessor_result_list = None
        self.tuner_finished_flag = None  # tuner get params -> false / tuner receive final_result -> true
        # finishing -> done # 也许会卡在这？？？

        # self.trial_id = trial_id
        # self.exp_id = exp_id

        # trial_dur ??


class SharedExpData:
    # static <-> dynamic 区别： 重复实验不需要dynamic，dynamic、是为了方便访问(区分于id_data频繁读写？？？)
    # c持久化的永远是过程....和最终结果
    # 1运行时会被其他组件读但不写 （得是对象才能共享
    # 2在实验结束后会被持久化 ???
    # trial_id (dynamic) / exp_dir ...? (static) / exp_start_time (dynamic) / stop_flag (dynamic)
    # id_data_dict

    def __init__(self, exp_config: ExperimentConfig):
        self._lock = threading.Lock()

        # __ 含义： 1并发读写(提供修改方法) 或者 2只写一次以后都只读
        self.__exps_dir = exp_config.exps_dir  #

        self.__id_data_dict = {}  # trial_id: trial_data
        self.__exp_start_time = time.time()
        self.__exp_stop_flag = False
        self.__exp_id = self.__get_unique_exp_id()  # 避免重复
        self.__create_dir_and_path()
        self.__exp_pid = os.getpid()  # ?

    # 下面是只能在类内部调用的方法 （只调用一次）（初始化时）
    def __get_unique_exp_id(self):
        exp_id_list = os.listdir(self.__exps_dir)
        while True:
            exp_id = get_uuid(8)
            if exp_id not in exp_id_list:
                return exp_id

    def __create_dir_and_path(self):
        exp_dir = self.get_exp_dir()
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)
        sub_dir_name_list = ["log", "db", "checkpoint", "trials"]
        for dir_name in sub_dir_name_list:
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)

    # 以下是 id_data_dict 的操作
    def __getitem__(self, item):
        with self._lock:
            return self.__id_data_dict[item]

    def __setitem__(self, key, value):
        with self._lock:
            self.__id_data_dict[key] = value

    def __len__(self):
        with self._lock:
            return len(self.__id_data_dict)

    def keys(self):
        with self._lock:
            return list(self.__id_data_dict.keys())  # copy

    def values(self):
        with self._lock:
            return list(self.__id_data_dict.values())  # copy

    def items(self) -> List[Tuple[str:SharedTrialData]]:
        with self._lock:
            return list(self.__id_data_dict.items())  # copy

    # 以下是一些只读成员的操作
    def get_exp_id(self):
        return self.__exp_id

    def get_exp_pid(self):
        return self.__exp_pid

    def get_exp_start_time(self):
        return self.__exp_start_time

    def get_exp_dir(self):
        return os.path.join(self.__exps_dir, self.__exp_id)

    def get_log_dir(self):
        return self.get_exp_dir() + "/log"

    def get_db_dir(self):
        return self.get_exp_dir() + "/db"

    def get_checkpoint_dir(self):
        return self.get_exp_dir() + "/checkpoint"

    def get_trials_dir(self):
        return self.get_exp_dir() + "/trials"

    # 以下是一些可读可写的操作
    def set_exp_stop(self):
        with self._lock:
            self.__exp_stop_flag = True

    def get_exp_stop_flag(self):
        with self._lock:
            return self.__exp_stop_flag


class Experiment:
    def __init__(self, exp_config: ExperimentConfig = None):
        self.logger = logging.getLogger(self.__class__.__name__)
        if type(exp_config) is not ExperimentConfig:
            raise ValueError("config should be ExperimentConfig")
        self.exp_config = exp_config
        self.exp_data: Optional[SharedExpData] = None

        self.tuner = None
        self.assessor = None
        self.allocator = None

        self.exp_pid = None  # 没有组件关心？
        self.id_thread_dict = None  # 只有exp_manager组件关心？-> trial_manager提供 必要的数据传输接口+(monitor操作)
        # -> thread内含trial_manager 所以只能在这

        self.heartbeat_thread = None

    def start(self):
        self.exp_data = SharedExpData(self.exp_config)
        self.id_thread_dict = {}

        self.tuner = Tuner(self.exp_config, self.exp_data)
        self.assessor = Assessor(self.exp_config, self.exp_data)
        self.allocator = Allocator(self.exp_config, self.exp_data)

        self.heartbeat_thread = threading.Thread(target=self._exp_heartbeat_thread)
        self.heartbeat_thread.start()
        self.heartbeat_thread.join()  # 手动停下只有kill 。。。 nohup ... &

    def get_unique_trial_id(self):
        l1 = os.listdir(self.exp_data.get_trials_dir())
        l2 = self.exp_data.keys()
        trial_id_set = list(set(l1).union(set(l2)))
        while True:
            trial_id = get_uuid(8)
            if trial_id not in trial_id_set:
                return trial_id

    def _exp_heartbeat_thread(self):
        # monitor 多rule的计算并行
        # assessor 和 tuner 本身会有 多trial的计算并行
        # （tuner应该还好 assessor可能会有问题 ？？-》全部采用轮询解决，（调用频率不高，可能还行，有待观察 实践检验！！！！
        # 并行的话 历史数据deepcopy感觉也会有额外开销
        # 实行了初步并行，至少不block训练的thread
        # 最多延迟发放quota/延迟wait转run/延迟run转done/真实模型一个epoch差距 -》 输出日志查看/凭感觉？？？
        # 部分_thread逻辑分别写入 tuner/assessor/trial_manager -> 独立的worker稳定的时间间隔？ lock还是trial级别
        # ->> id_trial_dict 也需要共享了？？？？ 以后再说吧。。。。。。。。。。。。
        def get_exp_dur():
            return int(time.time()) - self.exp_data.get_exp_start_time()

        def if_should_no_new_trial():
            if get_exp_dur > self.exp_config.max_exp_duration:
                return True
            if len(self.exp_data) >= self.exp_config.max_trial_number:
                return True
            return False

        def if_exp_stop_now():
            if self.exp_data.get_exp_stop_flag():
                return True
            if if_should_no_new_trial():
                for _trial_id, _trial_data in self.exp_data.items():
                    if _trial_data.status != TrialStatus.Done:
                        return False
                return True
            return False

        while True:
            # clean
            if if_exp_stop_now():
                # trial_thread 的最后清理 ...
                return
            # append waiting
            if not if_should_no_new_trial():
                cnt = 0
                for trial_id, trial_data in self.exp_data.items():
                    # trial_data: check status
                    trial_data.lock.acquire()
                    if trial_data.status != TrialStatus.Done:  # waiting running
                        cnt += 1
                    trial_data.lock.release()
                for i in range(self.exp_config.trial_concurrency - cnt):
                    trial_id = self.get_unique_trial_id()
                    self.exp_data[trial_id] = SharedTrialData()
            # run -> trial_manager
            for trial_id, trial_data in self.exp_data.items():
                trial_data.lock.acquire()
                if trial_data.status == TrialStatus.Waiting and trial_data.device_str is not None and trial_data.params is not None:
                    trial_data.status = TrialStatus.Running
                    target = self.exp_config.trial_func_config.ctor
                    args = (Trial(trial_id, trial_data, self.exp_config),)
                    trial_thread = threading.Thread(target=target, args=args)  ##################
                    trial_thread.start()
                    self.id_thread_dict[trial_id] = trial_thread
                trial_data.lock.release()
            # judge done
            for trial_id, trial_data in self.exp_data.items():
                trial_data.lock.acquire()
                # wait 4 assessor and tuner extract???
                if trial_data.status == TrialStatus.Running and self.id_thread_dict[trial_id].is_alive() is False:
                    trial_data.status = TrialStatus.Finishing
                if trial_data.status == TrialStatus.Finishing and trial_data.tuner_finished_flag is True and \
                    len(trial_data.assessor_result_list) == len(trial_data.interm_result_list):
                    trial_data.status = TrialStatus.Done
            time.sleep(1)

        while True:
            # epoch_quota / assessor_result
            for trial_id, trial_data in self.exp_data.items():
                trial_data.lock.acquire()
                if trial_data.status == TrialStatus.Running:
                    if trial_data.initial_result is not None:
                        self.tuner.post_initial_result(trial_id, trial_data.initial_result)
                    if len(trial_data.assessor_result_list) < len(trial_data.interm_result_list):
                        for interm_result in trial_data.interm_result_list[len(trial_data.assessor_result_list):]:
                            trial_data.assessor_result_list.append(
                                self.assessor.post_interm_result(trial_id, interm_result))
                trial_data.lock.release()

    def stop(self):
        self.exp_data.set_exp_stop()
        self.heartbeat_thread.join()

        self.tuner.heartbeat_thread.join()
        self.assessor.heartbeat_thread.join()
        self.allocator.heartbeat_thread.join()
        return

        # def resume(self):
    #     pass
    #
    # def view(self):
    #     pass
