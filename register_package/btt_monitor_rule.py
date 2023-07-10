import logging
import os
import queue
import threading
from abc import abstractmethod
from copy import deepcopy

import numpy as np
import torch
from scipy.stats import stats
from torch import nn

from btt_utils import get_module_name_nele_dict


class MonitorRuleBase:
    # 父类中的变量会在子类中共享！！！！！！！！
    def __init__(self, d_args):
        self.rule_name = d_args["rule_name"]
        self.logger = logging.getLogger(self.rule_name)
        pass

    @abstractmethod
    def record_metric(self, d_args):
        pass

    @abstractmethod
    def obtain_metric(self, d_args):
        pass
        # 三种模式 分别对应 user立即immediate intermediate等idx final等all


class ParallelMonitorRule(MonitorRuleBase):
    def __init__(self, d_args):
        super().__init__(d_args)
        self.task_queue = queue.Queue()
        self.result_dict = dict()
        self.result_dict_lock = threading.Lock()

        self.num_workers = 5 if "num_workers" not in d_args else d_args["num_workers"]
        self.worker_threads = []
        self.record_seq_idx = 0

        self.start_workers()

    def record_metric(self, d_args):
        d_args = self.before_record_metric(d_args)
        self.task_queue.put(tuple([self.record_seq_idx, d_args]))
        self.record_seq_idx += 1

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
        # return # join 需要return？

    def before_record_metric(self, d_args):
        return d_args
        # raise NotImplementedError

    def calc_metric_parallel(self, d_args):
        return d_args
        # raise NotImplementedError

    def start_workers(self):
        for i in range(self.num_workers):
            t = threading.Thread(target=self.worker_thread)
            # t.Daemon = True  #### !!!! Daemon 无法join
            t.start()
            self.worker_threads.append(t)

    def join_workers(self):
        self.logger.debug("join_workers: begin")
        for i in range(self.num_workers):
            self.task_queue.put(tuple([None, None]))
        for t in self.worker_threads:
            t.join()  # 卡住? okk!
        self.logger.debug("join_workers: done")

    def obtain_metric(self, d_args):
        mode = d_args["mode"]
        with self.result_dict_lock:
            self.result_dict = dict(sorted(self.result_dict.items(), key=lambda x: x[0]))
            keys = list(self.result_dict.keys())

        r = None
        self.logger.debug("obtain_metric: {}".format(d_args))
        if mode == "idx_immediate":
            result_idx = d_args["result_idx"]
            r = self.result_dict[keys[result_idx]] if result_idx < len(keys) else None
        elif mode == "idx_wait":
            result_idx = d_args["result_idx"]
            while True:
                if result_idx < len(keys):
                    r = self.result_dict[keys[result_idx]]
                    break
                os.system("sleep 1")
        elif mode == "all_wait":
            self.join_workers()  # 等待worker处理完其当前任务
            self.task_queue.join()
            self.result_dict = dict(sorted(self.result_dict.items(), key=lambda x: x[0]))
            r = list(self.result_dict.values())
        else:
            raise ValueError("unknown mode: {}".format(mode))
        d_args["result"] = r
        return self.after_obtain_metric(d_args)

    def after_obtain_metric(self, d_args):
        return d_args["result"]
        # raise NotImplementedError


class OnceMonitorRule(MonitorRuleBase):
    def __init__(self, d_args):
        super().__init__(d_args)
        self.once_flag = False
        self.result = None

    def record_metric(self, result):
        if self.once_flag:
            raise ValueError("OnceMonitorRule can only be used once")
        self.once_flag = True
        self.result = result

    def obtain_metric(self, d_args):
        return self.result


class PeriodMonitorRule(MonitorRuleBase):
    def __init__(self, d_args):
        super().__init__(d_args)
        self.result_list = []

    def record_metric(self, result):
        self.result_list.append(result)

    def obtain_metric(self, d_args):
        mode = d_args["mode"]
        result_idx = d_args["result_idx"]
        if mode == "idx_immediate":
            return self.result_list[result_idx] if result_idx < len(self.result_list) else None
        elif mode == "idx_wait":
            return self.result_list[result_idx] if result_idx < len(self.result_list) else None
        elif mode == "all_wait":
            return self.result_list
        else:
            raise ValueError("unknown mode: {}".format(mode))


class WeightStatisticsEpochMonitorRule(ParallelMonitorRule):
    # 收集一个epoch的所有batch的weight的统计信息
    def __init__(self, d_args):
        super().__init__(d_args)
        self.module_name_list = None
        self.default_module_type_list = [nn.Linear, nn.Conv2d, nn.Conv1d, nn.Conv3d, nn.RNN, nn.LSTM, nn.GRU]
        self.default_metric_suffix_list = ["mid", "rate0"]
        self.metric_prefix = d_args["metric_prefix"] if "metric_prefix" in d_args else None
        self.metric_suffix_list = self.default_metric_suffix_list \
            if "metric_suffix_list" not in d_args else d_args["metric_suffix_list"]
        self.module_type_list = self.default_module_type_list \
            if "module_type_list" not in d_args else d_args["module_type_list"]

    def before_record_metric(self, d_args):
        pre_model = d_args["model"]
        new_model = deepcopy(pre_model).cpu()
        pre_model_params = list(pre_model.parameters())
        new_model_params = list(new_model.parameters())
        for i in range(len(pre_model_params)):
            if pre_model_params[i].grad is not None and pre_model_params[i].requires_grad:
                new_model_params[i].grad = pre_model_params[i].grad.clone().detach().cpu()
        d_args["model"] = new_model

        ########

        mode = d_args["mode"]
        self.logger.debug("before_record_metric: {}".format(mode))
        if mode == "train_begin":
            model = d_args["model"]
            self.module_name_list = list(get_module_name_nele_dict(model, self.module_type_list).keys())
        elif mode == "train_iter_end":
            pass
        # elif mode == "train_epoch_end":
        #     pass  # 此处有点类似obtain逻辑 应该不处理
        else:
            raise ValueError("mode should be in ['train_begin', 'train_iter_end']")
        return d_args

    def calc_metric_parallel(self, d_args):
        # mode, model, max_nb_batch, max_nb_epoch, module_type_list
        mode = d_args["mode"]
        model = d_args["model"]
        self.logger.debug("calc_metric_parallel begin: {}".format(mode))

        if mode == "train_begin":
            pass
        elif mode == "train_iter_end":
            # module_name->module_idx和metric_suffix->metric_idx的映射 为了方便后续按照epoch聚合
            single_batch_module_metric_2da = np.zeros((len(self.module_name_list), len(self.metric_suffix_list)))
            for (module_name, module) in model.named_modules():
                # 判断是否是基类
                if not isinstance(module, tuple(self.default_module_type_list)):
                    continue
                for (param_name, param) in module.named_parameters():
                    if "weight" not in param_name:
                        continue
                    metric_idx = 0
                    for suffix in self.metric_suffix_list:
                        metric_name = self.metric_prefix + "_" + suffix
                        module_idx = self.module_name_list.index(module_name)
                        if self.metric_prefix == "weight_val":
                            tmp = param.detach().cpu().numpy().flatten()
                        elif self.metric_prefix == "weight_val_abs":
                            tmp = np.abs(param.detach().cpu().numpy().flatten())
                        elif self.metric_prefix == "weight_grad":  # no detach
                            tmp = param.grad.cpu().numpy().flatten()
                        elif self.metric_prefix == "weight_grad_abs":
                            tmp = np.abs(param.grad.cpu().numpy().flatten())
                        # elif self.metric_prefix == "module_feature":
                        #     tmp = module.feature.detach().cpu().numpy().flatten()
                        else:
                            raise ValueError("metric_prefix should be in ['weight_val', 'weight_val_abs']")
                        if suffix == "avg":
                            single_batch_module_metric_2da[module_idx][metric_idx] = np.mean(tmp)
                        elif suffix == "var":
                            single_batch_module_metric_2da[module_idx][metric_idx] = np.var(tmp)
                        elif suffix == "mid":
                            single_batch_module_metric_2da[module_idx][metric_idx] = np.median(tmp)
                        elif suffix == "max":
                            single_batch_module_metric_2da[module_idx][metric_idx] = np.max(tmp)
                        elif suffix == "min":
                            single_batch_module_metric_2da[module_idx][metric_idx] = np.min(tmp)
                        elif suffix == "upper":
                            single_batch_module_metric_2da[module_idx][metric_idx] = np.percentile(tmp, 75)
                        elif suffix == "lower":
                            single_batch_module_metric_2da[module_idx][metric_idx] = np.percentile(tmp, 25)
                        elif suffix == "skew":
                            single_batch_module_metric_2da[module_idx][metric_idx] = stats.skew(tmp)
                        elif suffix == "kurt":
                            single_batch_module_metric_2da[module_idx][metric_idx] = stats.kurtosis(tmp)
                        elif suffix == "rate0":
                            single_batch_module_metric_2da[module_idx][metric_idx] = np.sum(tmp == 0) / tmp.size
                        else:
                            raise ValueError("metric_suffix should be in ['avg', 'var', 'mid', 'max', 'min', "
                                             "'upper', 'lower', 'skew', 'kurt', 'rate0']")
                        metric_idx += 1
            return single_batch_module_metric_2da
        # elif mode == "train_epoch_end":
        #     pass  # 此处有点类似obtain逻辑 应该不处理
        else:
            raise ValueError("mode should be in ['train_begin', 'train_iter_end']")

    def after_obtain_metric(self, d_args):
        mode = d_args["mode"]
        result = d_args["result"]
        if result is None:
            return None

        self.logger.debug("after_obtain_metric: {}".format(mode))
        if mode == "idx_immediate":
            return result
        elif mode == "idx_wait":
            return result
        elif mode == "all_wait":
            result_list = result
            nb_batch = len(result_list)
            shape = (nb_batch, len(self.module_name_list), len(self.metric_suffix_list))
            batch_module_metric_3da = np.zeros(shape)
            for i in range(nb_batch):
                batch_module_metric_3da[i] = result_list[i]
            single_epoch_module_metric_2da = np.mean(batch_module_metric_3da, axis=0)
            return single_epoch_module_metric_2da
        else:
            raise ValueError("mode should be in ['idx_immediate', 'idx_wait', 'all_wait']")


class WeightStatisticsMonitorRule(MonitorRuleBase):
    def __init__(self, d_args):
        super().__init__(d_args)
        self.module_name_list = None
        self.epoch_rule_instance_list = []
        self.default_module_type_list = [nn.Linear, nn.Conv2d, nn.Conv1d, nn.Conv3d, nn.RNN, nn.LSTM, nn.GRU]
        self.default_metric_suffix_list = ["mid", "rate0"]
        self.init_args = None
        self.train_begin_args_epoch = None
        self.next_train_begin_flag = False
        self.single_epoch_module_metric_2da_list = []

        self.metric_prefix = d_args["metric_prefix"] if "metric_prefix" in d_args else None
        self.metric_suffix_list = self.default_metric_suffix_list \
            if "metric_suffix_list" not in d_args else d_args["metric_suffix_list"]
        self.module_type_list = self.default_module_type_list \
            if "module_type_list" not in d_args else d_args["module_type_list"]
        self.init_args = d_args

    def record_metric(self, d_args):
        mode = d_args["mode"]
        model = d_args["model"]

        self.logger.debug("record_metric: {}".format(mode))
        if mode == "train_begin":
            self.module_name_list = list(get_module_name_nele_dict(model, self.module_type_list).keys())
            self.train_begin_args_epoch = d_args
            self.next_train_begin_flag = True
        elif mode == "train_iter_end":
            if self.next_train_begin_flag:
                self.next_train_begin_flag = False
                d = self.init_args.copy()
                d["rule_name"] = d["rule_name"] + "_epoch_" + str(len(self.epoch_rule_instance_list))
                self.epoch_rule_instance_list.append(WeightStatisticsEpochMonitorRule(d))
                self.epoch_rule_instance_list[-1].record_metric(self.train_begin_args_epoch)
            self.epoch_rule_instance_list[-1].record_metric(d_args)
        elif mode == "train_epoch_end":
            self.next_train_begin_flag = True
        else:
            raise ValueError("mode should be in ['train_begin', 'train_iter_end', 'train_epoch_end']")

    def obtain_metric(self, d_args):
        mode = d_args["mode"]
        self.logger.debug("obtain_metric: {}".format(mode))
        if mode == "idx_immediate":
            result_idx = d_args["result_idx"] if "result_idx" in d_args else None
            return self.epoch_rule_instance_list[result_idx].obtain_metric(d_args)
        elif mode == "idx_wait":
            result_idx = d_args["result_idx"] if "result_idx" in d_args else None
            d_args_ = deepcopy(d_args)
            d_args_["result_idx"] = None
            d_args_["mode"] = "all_wait"
            r = self.epoch_rule_instance_list[result_idx].obtain_metric(d_args_)
            self.single_epoch_module_metric_2da_list.append(r)
            return r
        elif mode == "all_wait":
            tmp = self.epoch_rule_instance_list[0]
            shape = (len(self.epoch_rule_instance_list), len(tmp.module_name_list), len(tmp.metric_suffix_list))
            epoch_module_metric_3da = np.zeros(shape)
            # 可能需要等intermediate_metric_rule的所有结果都出来才能计算
            for i in range(len(self.epoch_rule_instance_list)):
                epoch_module_metric_3da[i] = self.single_epoch_module_metric_2da_list[i]
            return epoch_module_metric_3da


class NanInfMonitorRule(MonitorRuleBase):
    # eeeeeeeeeeeee
    def __init__(self, d_args):
        super().__init__(d_args)
        self.nan_inf_list = []

    def record_metric(self, kv_args):
        nan_inf_flag = False
        model = kv_args["model"]
        for (module_name, module) in model.named_modules():
            for (param_name, param) in module.named_parameters():
                if torch.isnan(param).any():
                    print("Nan in param: ", module_name, param_name)
                    nan_inf_flag = True
                if torch.isinf(param).any():
                    print("Inf in param: ", module_name, param_name)
                    nan_inf_flag = True
                if param.grad is not None:
                    if torch.isnan(param.grad).any():
                        print("Nan in param.grad: ", module_name, param_name)
                        nan_inf_flag = True
                    if torch.isinf(param.grad).any():
                        print("Inf in param.grad: ", module_name, param_name)
                        nan_inf_flag = True
        self.nan_inf_list.append(nan_inf_flag)

    def obtain_metric(self, kwargs):
        return self.nan_inf_list


class NotImplementMonitorRule(MonitorRuleBase):

    def record_metric(self, kwargs):
        return

    def obtain_metric(self, kwargs):
        return "nothing"
