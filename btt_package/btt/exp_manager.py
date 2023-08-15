import json
import os
import threading
from typing import Optional, Dict, Any, List

from .utils import get_uuid, get_package_abs_dir


class AlgorithmConfig:
    # for: monitor_rule tuner_hpo assessor_indicator
    # btt的monitor/tuner/assessor其实都只有一个 只是具体可能存在多个(也不一定同时用)的rule/hpo/indicator！
    name: Optional[str] = None
    module_path: Optional[os.PathLike, str] = None  # "/".split
    class_name: Optional[str] = None
    init_args: Optional[Dict[str, Any]] = None

    def canonicalize(self):
        if self.name is None:
            self.name = self.class_name
        if self.init_args is None:
            self.init_args = {}


class TunerHpoConfig(AlgorithmConfig):
    # max_trial_number max_exp_duration
    seed = None
    # 优先级别：list前面的优先级高，之后exp_dur优先级高于trial_num
    exp_dur_not_trial_num: bool = None  # True: exp_dur False: trial_num # 考虑到batchtuner
    start_ratio: Optional[float] = None
    end_ratio: Optional[float] = None

    # default: Optional[bool] = False # 0-1默认default

    def canonicalize(self):
        def judge_in_range(val_list, exp_min, exp_max, exp_type):
            for val in val_list:
                if type(val) is not exp_type:
                    raise ValueError("val should be " + str(exp_type))
                if val < exp_min or val > exp_max:
                    raise ValueError("val should be in range [" + str(exp_min) + "," + str(exp_max) + "]")

        super().canonicalize()
        package_dir = get_package_abs_dir()
        self.seed = 529 if self.seed is None else self.seed
        self.exp_dur_not_trial_num = True if self.exp_dur_not_trial_num is None else self.exp_dur_not_trial_num
        if self.start_ratio is not None or self.end_ratio is not None:
            judge_in_range([self.start_ratio, self.end_ratio], 0, 1, float)
            self.start_ratio = 0.0 if self.start_ratio is None else self.start_ratio
            self.end_ratio = 1.0 if self.end_ratio is None else self.end_ratio
        self.module_path = os.path.join(package_dir, "tuner/tuner_hpo") \
            if self.module_path is None else self.module_path


class MonitorRuleConfig(AlgorithmConfig):
    intermediate_report: Optional[bool] = None
    intermediate_default: Optional[bool] = None
    final_report: Optional[bool] = None
    final_default: Optional[bool] = None

    def canonicalize(self):
        super().canonicalize()
        package_dir = get_package_abs_dir()
        self.module_path = os.path.join(package_dir, "monitor/monitor_rule") \
            if self.module_path is None else self.module_path
        self.intermediate_report = False if self.intermediate_report is None else self.intermediate_report
        self.intermediate_default = False if self.intermediate_default is None else self.intermediate_default
        self.final_report = False if self.final_report is None else self.final_report
        self.final_default = False if self.final_default is None else self.final_default


class AssessorIndicatorConfig(AlgorithmConfig):
    # exp_dur
    # start_ratio: Optional[float] = None
    # end_ratio: Optional[float] = None #没必要徒增烦恼
    # start/end的ratio跟indicator自身内容是相关的 跟实验性质其实无关

    def canonicalize(self):
        super().canonicalize()
        package_dir = get_package_abs_dir()
        self.module_path = os.path.join(package_dir, "assessor/assessor_indicator") \
            if self.module_path is None else self.module_path


class ExperimentConfig:
    # 给用户 方便填写
    # exp_uuid: str = None # 用户不填
    exp_name: Optional[str] = None
    exp_description: Optional[str] = None
    trial_concurrency: int = None
    trial_gpu_number: int = None
    max_trial_number: int = None
    max_exp_duration: Optional[int, str] = None  # int -> seconds
    tuner_config: List[TunerHpoConfig] = None  # 本质是 tuner_hpo_config_list
    monitor_config: List[MonitorRuleConfig] = None
    assessor_config: List[AssessorIndicatorConfig] = None
    btt_exp_dir: Optional[os.PathLike] = None

    def __init__(self, exp_uuid=None, btt_exp_dir=None):
        self.config_file_name = "exp_config.json"  #### 用户不能改的
        if type(exp_uuid) is str:
            self.exp_uuid = exp_uuid
            self.btt_exp_dir = btt_exp_dir if btt_exp_dir is not None else "./btt_experiments"
            self.exp_config = self.load_exp_config()
        pass

    def canonicalize(self):
        self.exp_name = "default_exp_name" if self.exp_name is None else self.exp_name
        self.exp_description = "default_exp_description" if self.exp_description is None else self.exp_description
        self.trial_concurrency = 4 if self.trial_concurrency is None else self.trial_concurrency
        self.trial_gpu_number = 0 if self.trial_gpu_number is None else self.trial_gpu_number
        self.max_trial_number = 1000 if self.max_trial_number is None else self.max_trial_number
        self.max_exp_duration = 12 * 60 * 60 if self.max_exp_duration is None else self.max_exp_duration
        if self.tuner_config is None:
            raise ValueError("tuner_config should not be None")
        self.monitor_config = [] if self.monitor_config is None else self.monitor_config
        self.assessor_config = [] if self.assessor_config is None else self.assessor_config
        self.btt_exp_dir = "./btt_experiments" if self.btt_exp_dir is None else self.btt_exp_dir

    def load_exp_config(self):
        exp_dir = os.path.join(self.btt_exp_dir, self.exp_uuid)
        exp_config_path = os.path.join(exp_dir, self.config_file_name)
        with open(exp_config_path, "r") as f:
            exp_config = json.load(f)
        return exp_config

    def save_exp_config(self):
        exp_config_path = os.path.join(self.btt_exp_dir, self.exp_uuid, self.config_file_name)
        with open(exp_config_path, "w") as f:
            json.dump(self.exp_config, f)


class BttExperimentManager:
    def __init__(self, exp_config_instance):
        # start stop resume view
        # self.exp_config_instance = ExperimentConfig(exp_config_args)  # 用来保存ok 实际使用可以直接哟过exp_manager
        self.exp_id = get_uuid(8)
        if type(exp_config_instance) is str:
            self.exp_uuid = exp_config_instance
            self.btt_exp_dir = "./btt_experiments" if "btt_exp_dir" not in exp_config_instance else exp_config_instance[
                "btt_exp_dir"]
            self.exp_config = self.load_exp_config()

        self.exp_config = exp_config_instance
        self.exp_name = exp_config_instance["exp_name"]
        self.exp_description = exp_config_instance["exp_description"]

        self.btt_exp_dir = exp_config_instance[
            "btt_exp_dir"] if "btt_exp_dir" in exp_config_instance else "./btt_experiments"
        self.exp_config_path = self.btt_exp_dir + "/exp_config.json"

        self.exp_dir = self.btt_exp_dir + "/" + self.exp_id
        self.log_dir = self.exp_dir + "/log"
        self.db_dir = self.exp_dir + "/db"
        self.checkpoint_dir = self.exp_dir + "/checkpoint"
        self.trials_dir = self.exp_dir + "/trials"
        self.create_dir_and_path()

        # self.exp_db = self.create_exp_db() # 暂时不需要db？可以最后再加入
        self.trial_concurrency = exp_config_instance["trial_concurrency"]
        self.trial_gpu_number = exp_config_instance[
            "trial_gpu_number"] if "trial_gpu_number" in exp_config_instance else 0
        self.max_trial_number = exp_config_instance[
            "max_trial_number"] if "max_trial_number" in exp_config_instance else 1000
        self.max_exp_duration = exp_config_instance[
            "max_exp_duration"] if "max_exp_duration" in exp_config_instance else "12h"
        # self.space_path = config_args["space_path"] # 逐步传入，第一次report时确定

        self.monitor_config = exp_config_instance["monitor_config"] if "monitor_config" in exp_config_instance else None
        self.tuner_config = exp_config_instance["tuner_config"] if "tuner_config" in exp_config_instance else None
        self.assessor_config = exp_config_instance[
            "assessor_config"] if "assessor_config" in exp_config_instance else None

    def start_workers(self):
        for i in range(self.num_workers):
            t = threading.Thread(target=self.worker_thread)
            # t.Daemon = True  #### !!!! Daemon 无法join
            t.start()
            self.worker_threads.append(t)

    def worker_thread(self):
        while True:
            try:
                record_idx, d_args = self.task_queue.get(block=True, timeout=1)  # block 问题在于可能block带锁？
            except queue.Empty:
                continue
            if d_args is None:
                self.task_queue.task_done()
                break
            result = self.calc_metric_parallel(d_args)
            if result is not None:
                with self.result_dict_lock:
                    self.result_dict[record_idx] = result
                    self.result_dict = dict(sorted(self.result_dict.items(), key=lambda x: x[0]))
            self.task_queue.task_done()
        self.logger.debug("worker_thread: done")

    def create_dir_and_path(self):
        if not os.path.exists(self.exp_dir):
            os.makedirs(self.exp_dir)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if not os.path.exists(self.db_dir):
            os.makedirs(self.db_dir)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if not os.path.exists(self.trials_dir):
            os.makedirs(self.trials_dir)

    def start(self):
        pass

    def stop(self):
        pass

    def resume(self):
        pass

    def view(self):
        pass
