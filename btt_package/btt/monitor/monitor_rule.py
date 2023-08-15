import logging
import os
import queue
import threading
from abc import abstractmethod
from copy import deepcopy

import numpy as np
import torch
from torch import nn

from ..utils import get_module_name_nele_dict, get_module_id_name_dict, calc_array_statistic, \
    RecordMode, ObtainMode


class MonitorRuleBase:
    # 父类中的变量会在子类中共享！！！！！！！！
    def __init__(self, d_args):
        self.rule_name = d_args["rule_name"]
        self.logger = logging.getLogger(self.rule_name)
        # self.active_mode_list = d_args["active_mode_list"]
        # self.logger.setLevel(logging.DEBUG)
        pass

    @abstractmethod
    def record_metric(self, d_args):
        raise NotImplementedError

    @abstractmethod
    def obtain_metric(self, d_args):
        raise NotImplementedError


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
        if d_args is None:
            return  # none -> 不记录
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
        if mode == ObtainMode.IdxImmediate:
            result_idx = d_args["result_idx"]
            r = self.result_dict[keys[result_idx]] if result_idx < len(keys) else None
        elif mode == ObtainMode.IdxWait:
            result_idx = d_args["result_idx"]
            while True:
                if result_idx < len(keys):
                    r = self.result_dict[keys[result_idx]]
                    break
                os.system("sleep 1")
        elif mode == ObtainMode.AllWait:
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


class OnceMonitorRuleBase(MonitorRuleBase):
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


class PeriodMonitorRuleBase(MonitorRuleBase):
    def __init__(self, d_args):
        super().__init__(d_args)
        self.result_list = []

    def record_metric(self, result):
        self.result_list.append(result)

    def obtain_metric(self, d_args):
        mode = d_args["mode"]
        result_idx = d_args["result_idx"] if "result_idx" in d_args else None
        if mode == ObtainMode.IdxImmediate:
            return self.result_list[result_idx] if result_idx < len(self.result_list) else None
        elif mode == ObtainMode.IdxWait:
            return self.result_list[result_idx] if result_idx < len(self.result_list) else None
        elif mode == ObtainMode.AllWait:
            return self.result_list
        else:
            raise ValueError("unknown mode: {}".format(mode))


class WeightStatisticsEpochMonitorRule(ParallelMonitorRule):
    # 收集一个epoch的所有batch的weight的统计信息
    def __init__(self, d_args):
        super().__init__(d_args)
        self.module_name_list = d_args["module_name_list"]
        self.metric_prefix = d_args["metric_prefix"]
        self.metric_suffix_list = d_args["metric_suffix_list"]
        self.module_type_list = d_args["module_type_list"]

        self.max_nb_calc_batch = d_args["max_nb_calc_batch"]

        self.calc_batch_idx = None
        self.stop_flag = None

    def before_record_metric(self, d_args):
        def get_tensor_list(model: nn.Module):
            tensor_list = [None] * len(self.module_name_list)
            for (module_name, module) in model.named_modules():
                if not isinstance(module, tuple(self.module_type_list)):
                    continue
                for (param_name, param) in module.named_parameters():  # ?
                    if "weight" not in param_name:
                        continue
                    # detach() 本身并不会新建tensor！（共享内存）
                    module_idx = self.module_name_list.index(module_name)

                    if "weight_val" in self.metric_prefix:
                        tensor = param.detach().clone().cpu().flatten()
                    elif "weight_grad" in self.metric_prefix:
                        tensor = param.grad.detach().clone().cpu().flatten()
                    else:
                        raise ValueError("unknown metric_prefix: {}".format(self.metric_prefix))
                    if "_abs" in self.metric_prefix:
                        tensor = torch.abs(tensor)
                    tensor_list[module_idx] = tensor
            return tensor_list

        mode = d_args["mode"]
        self.logger.debug("before_record_metric: {}".format(mode))
        if mode == RecordMode.EpochTrainBegin:
            self.calc_batch_idx = 0
            self.stop_flag = False
        elif mode == RecordMode.TrainIterEnd:
            if self.stop_flag:
                self.calc_batch_idx += 1
                d_args["stop_flag"] = True
                return d_args
            if self.calc_batch_idx + 1 == self.max_nb_calc_batch:
                self.logger.debug("max_nb_calc_batch: {} stop".format(self.max_nb_calc_batch))
                self.stop_flag = True
            self.calc_batch_idx += 1
            d_args["tensor_list"] = get_tensor_list(d_args["model"])
        else:
            raise ValueError("unknown mode: {}".format(mode))
        return d_args

    def calc_metric_parallel(self, d_args):
        mode = d_args["mode"]
        self.logger.debug("calc_metric_parallel begin: {}".format(mode))

        if "stop_flag" in d_args and d_args["stop_flag"]:
            return

        if mode == RecordMode.EpochTrainBegin:
            pass
        elif mode == RecordMode.TrainIterEnd:
            single_batch_module_metric_2da = np.zeros((len(self.module_name_list), len(self.metric_suffix_list)))
            tensor_list = d_args["tensor_list"]
            for module_idx, tensor in enumerate(tensor_list):
                array = tensor.numpy()
                for metric_idx, suffix in enumerate(self.metric_suffix_list):
                    metric_name = self.metric_prefix + "_" + suffix
                    single_batch_module_metric_2da[module_idx, metric_idx] = calc_array_statistic(array, suffix)
            return single_batch_module_metric_2da
        else:
            raise ValueError("unknown mode: {}".format(mode))

    def after_obtain_metric(self, d_args):
        mode = d_args["mode"]
        result = d_args["result"]
        if result is None:
            return None

        self.logger.debug("after_obtain_metric: {}".format(mode))
        if mode == ObtainMode.IdxImmediate:
            return result
        elif mode == ObtainMode.IdxWait:
            return result
        elif mode == ObtainMode.AllWait:
            result_list = result
            nb_batch = len(result_list)
            shape = (nb_batch, len(self.module_name_list), len(self.metric_suffix_list))
            batch_module_metric_3da = np.zeros(shape)
            for i in range(nb_batch):
                batch_module_metric_3da[i] = result_list[i]
            single_epoch_module_metric_2da = np.mean(batch_module_metric_3da, axis=0)
            return single_epoch_module_metric_2da
        else:
            raise ValueError("unknown mode: {}".format(mode))


class WeightStatisticsMonitorRule(MonitorRuleBase):
    def __init__(self, d_args):
        super().__init__(d_args)
        self.module_name_list = None
        self.epoch_rule_instance_list = []
        self.single_epoch_module_metric_2da_list = []

        self.metric_prefix = d_args["metric_prefix"] if "metric_prefix" in d_args else None
        self.metric_suffix_list = ["mid", "rate0"] \
            if "metric_suffix_list" not in d_args else d_args["metric_suffix_list"]
        self.module_type_list = [nn.Linear, nn.Conv2d, nn.Conv1d, nn.Conv3d, nn.RNN, nn.LSTM, nn.GRU] \
            if "module_type_list" not in d_args else d_args["module_type_list"]
        self.calc_batch_ratio = 0.01 if "calc_batch_ratio" in d_args else d_args["calc_batch_ratio"]

        self.max_nb_batch = None
        self.max_nb_calc_batch = None

    def record_metric(self, d_args):
        def append_new_epoch_instance():
            init_args = {
                "rule_name": self.rule_name + "_epoch_" + str(len(self.epoch_rule_instance_list)),
                "module_name_list": self.module_name_list,
                "metric_prefix": self.metric_prefix,
                "metric_suffix_list": self.metric_suffix_list,
                "module_type_list": self.module_type_list,
                "max_nb_calc_batch": self.max_nb_calc_batch,
            }
            self.epoch_rule_instance_list.append(WeightStatisticsEpochMonitorRule(init_args))

        mode = d_args["mode"]
        self.logger.debug("record_metric: {}".format(mode))
        if mode == RecordMode.Begin:
            model = d_args["model"]
            self.module_name_list = list(get_module_name_nele_dict(model, self.module_type_list).keys())
            max_nb_batch = d_args["max_nb_batch"] if "max_nb_batch" in d_args else None  # 100?
            self.max_nb_calc_batch = max(3, int(max_nb_batch * self.calc_batch_ratio))
        elif mode == RecordMode.EpochTrainBegin:
            # model = d_args["model"]  ### pop del no! 所有rule公用！！！
            append_new_epoch_instance()
            self.epoch_rule_instance_list[-1].record_metric(d_args)  # simple init
        elif mode == RecordMode.TrainIterEnd:
            self.epoch_rule_instance_list[-1].record_metric(d_args)  # -> train begin
        else:
            return None

    def obtain_metric(self, d_args):
        mode = d_args["mode"]
        self.logger.debug("obtain_metric: {}".format(mode))
        if mode == ObtainMode.IdxImmediate:
            result_idx = d_args["result_idx"] if "result_idx" in d_args else None
            return self.epoch_rule_instance_list[result_idx].obtain_metric(d_args)
        elif mode == ObtainMode.IdxWait:
            result_idx = d_args["result_idx"] if "result_idx" in d_args else None
            d_args_ = deepcopy(d_args)
            d_args_["result_idx"] = None
            d_args_["mode"] = ObtainMode.AllWait
            r = self.epoch_rule_instance_list[result_idx].obtain_metric(d_args_)
            self.single_epoch_module_metric_2da_list.append(r)
            return r
        elif mode == ObtainMode.AllWait:
            tmp = self.epoch_rule_instance_list[0]
            shape = (len(self.epoch_rule_instance_list), len(tmp.module_name_list), len(tmp.metric_suffix_list))
            epoch_module_metric_3da = np.zeros(shape)
            # 可能需要等intermediate_metric_rule的所有结果都出来才能计算
            for i in range(len(self.epoch_rule_instance_list)):
                epoch_module_metric_3da[i] = self.single_epoch_module_metric_2da_list[i]
            return epoch_module_metric_3da
        else:
            raise ValueError("unknown mode: {}".format(mode))


class FeatureStatisticsEpochMonitorRule(ParallelMonitorRule):
    def __init__(self, d_args):
        super().__init__(d_args)
        # metric_prefix -> ["feat_val","feat_grad"] in or out ???
        self.module_name_list = d_args["module_name_list"]
        self.metric_prefix = d_args["metric_prefix"]
        self.metric_suffix_list = d_args["metric_suffix_list"]
        self.module_type_list = d_args["module_type_list"]

        self.max_nb_calc_batch = d_args["max_nb_calc_batch"]
        self.module_id_name_dict = d_args["module_id_name_dict"]

        self.calc_batch_idx = None
        self.stop_flag = None

        self.hook_handle_list = None

        self.tensor_list = None

    def before_record_metric(self, d_args):
        def register_hook(model: nn.Module):
            def forward_hook_get_feature_value_in(module: nn.Module, feature_value_in, feature_value_out):
                if not module.training:
                    return
                module_id = id(module)
                module_idx = self.module_name_list.index(self.module_id_name_dict[module_id])
                # array = feature_value_in[0].detach().cpu().numpy().flatten()  # time ...
                # tensor = feature_value_in[0].detach().cpu() # 因为并行计算所以cpu 但是.cpu耗时间？？？
                tensor = feature_value_in[0].detach().flatten()  # time ...
                self.logger.debug("forward_hook: module_idx: {}, tensor.shape: {}".format(module_idx, tensor.shape))
                self.tensor_list[module_idx] = tensor
                return

            def backward_hook_get_feature_gradient_out(module: nn.Module, feature_grad_in, feature_grad_out):
                if not module.training:
                    return
                module_id = id(module)
                module_idx = self.module_name_list.index(self.module_id_name_dict[module_id])
                tensor = feature_grad_out[0].detach().flatten()  # time ...
                self.logger.debug("backward_hook: module_idx: {}, tensor.shape: {}".format(module_idx, tensor.shape))
                self.tensor_list[module_idx] = tensor
                return

            self.hook_handle_list = []
            for module in model.modules():
                if type(module) not in self.module_type_list:
                    continue
                if self.metric_prefix == "feature_val_in":
                    h = module.register_forward_hook(forward_hook_get_feature_value_in)
                    self.hook_handle_list.append(h)
                elif self.metric_prefix == "feature_grad_out":
                    h = module.register_full_backward_hook(backward_hook_get_feature_gradient_out)
                    self.hook_handle_list.append(h)
                else:
                    raise ValueError("metric_prefix should be in ['feature_val_in', 'feature_grad_out']")
                self.hook_handle_list.append(h)

        def unregister_hook():
            if self.stop_flag:
                return
            for h in self.hook_handle_list:
                h.remove()
            self.stop_flag = True

        mode = d_args["mode"]
        if mode == RecordMode.EpochTrainBegin:
            register_hook(d_args["model"])
            self.tensor_list = [None] * len(self.module_name_list)
            self.calc_batch_idx = 0
            self.stop_flag = False
        elif mode == RecordMode.EpochTrainEnd:
            unregister_hook()
        elif mode == RecordMode.TrainIterEnd:
            if self.stop_flag:
                self.calc_batch_idx += 1
                d_args["stop_flag"] = True
                return d_args  # 不再计算
            if self.calc_batch_idx + 1 == self.max_nb_calc_batch:  # e.g. 2 + 1 = 3
                self.logger.debug("max_nb_calc_batch: {} stop".format(self.max_nb_calc_batch))
                unregister_hook()  # 断绝之后的hook，本次依旧正常计算
                self.stop_flag = True
            self.calc_batch_idx += 1
            d_args["module_tensor_list"] = self.tensor_list  # 类似record操作
        else:
            raise ValueError("unknown mode: {}".format(mode))
        return d_args

    def calc_metric_parallel(self, d_args):
        mode = d_args["mode"]
        self.logger.debug("calc_metric_parallel begin: {}".format(mode))

        if "stop_flag" in d_args and d_args["stop_flag"]:
            return

        if mode == RecordMode.EpochTrainBegin:
            pass
        elif mode == RecordMode.EpochTrainEnd:
            pass
        elif mode == RecordMode.TrainIterEnd:
            module_tensor_list = d_args["module_tensor_list"]
            single_batch_module_metric_2da = np.zeros((len(module_tensor_list), len(self.metric_suffix_list)))
            for module_idx, tensor in enumerate(module_tensor_list):
                array = tensor.cpu().numpy()
                for metric_idx, metric_suffix in enumerate(self.metric_suffix_list):
                    statistic = calc_array_statistic(array, metric_suffix)
                    single_batch_module_metric_2da[module_idx][metric_idx] = statistic
            return single_batch_module_metric_2da
        else:
            raise ValueError("unknown mode: {}".format(mode))

    def after_obtain_metric(self, d_args):
        mode = d_args["mode"]
        result = d_args["result"]
        if result is None:
            return None

        self.logger.debug("after_obtain_metric: {}".format(mode))
        if mode == ObtainMode.IdxImmediate:
            return result
        elif mode == ObtainMode.IdxWait:
            return result
        elif mode == ObtainMode.AllWait:
            result_list = result
            nb_calc_batch = len(result_list)
            shape = (nb_calc_batch, len(self.module_name_list), len(self.metric_suffix_list))
            batch_module_metric_3da = np.zeros(shape)
            for i in range(nb_calc_batch):
                batch_module_metric_3da[i] = result_list[i]
            single_epoch_module_metric_2da = np.mean(batch_module_metric_3da, axis=0)
            return single_epoch_module_metric_2da
        else:
            raise ValueError("mode {} not supported".format(mode))


class FeatureStatisticsMonitorRule(MonitorRuleBase):
    def __init__(self, d_args):
        super().__init__(d_args)
        self.module_name_list = None
        self.epoch_rule_instance_list = []
        self.default_module_type_list = [nn.Linear, nn.Conv2d, nn.Conv1d, nn.Conv3d, nn.RNN, nn.LSTM, nn.GRU]
        self.default_metric_suffix_list = ["mid", "rate0"]
        self.begin_args_for_epoch = None
        self.next_train_begin_flag = False
        self.single_epoch_module_metric_2da_list = []

        self.metric_prefix = d_args["metric_prefix"] if "metric_prefix" in d_args else None
        self.metric_suffix_list = self.default_metric_suffix_list \
            if "metric_suffix_list" not in d_args else d_args["metric_suffix_list"]
        self.module_type_list = self.default_module_type_list \
            if "module_type_list" not in d_args else d_args["module_type_list"]
        self.calc_batch_ratio = d_args["calc_batch_ratio"] if "calc_batch_ratio" in d_args else None

        self.max_nb_batch = None
        self.max_nb_calc_batch = None
        self.module_id_name_dict = None

    def record_metric(self, d_args):
        def append_new_epoch_instance():
            init_args4epoch = {
                "rule_name": self.rule_name + "_epoch_" + str(len(self.epoch_rule_instance_list)),
                "module_name_list": self.module_name_list,
                "metric_prefix": self.metric_prefix,
                "metric_suffix_list": self.metric_suffix_list,
                "module_type_list": self.module_type_list,
                "max_nb_calc_batch": self.max_nb_calc_batch,
                "module_id_name_dict": self.module_id_name_dict,
            }
            self.epoch_rule_instance_list.append(FeatureStatisticsEpochMonitorRule(init_args4epoch))

        mode = d_args["mode"]
        self.logger.debug("record_metric: {}".format(mode))
        if mode == RecordMode.Begin:
            model = d_args["model"]
            self.module_name_list = list(get_module_name_nele_dict(model, self.module_type_list).keys())
            self.module_id_name_dict = get_module_id_name_dict(model, self.module_type_list)
            self.begin_args_for_epoch = deepcopy(d_args)  # model 假的 无法register
            self.begin_args_for_epoch["mode"] = "begin"
            self.max_nb_batch = d_args["max_nb_batch"] if "max_nb_batch" in d_args else None  # 100?
            self.max_nb_calc_batch = max(int(self.calc_batch_ratio * self.max_nb_batch), 3)
        elif mode == RecordMode.EpochTrainBegin:
            append_new_epoch_instance()  # -> begin
            self.epoch_rule_instance_list[-1].record_metric(d_args)  # -> train begin
        elif mode == RecordMode.EpochTrainEnd:
            self.epoch_rule_instance_list[-1].record_metric(d_args)
        elif mode == RecordMode.TrainIterBegin:
            pass
        elif mode == RecordMode.TrainIterEnd:
            self.epoch_rule_instance_list[-1].record_metric(d_args)
        else:
            return None

    def obtain_metric(self, d_args):
        mode = d_args["mode"]
        self.logger.debug("obtain_metric: {}".format(mode))
        if mode == ObtainMode.IdxImmediate:
            result_idx = d_args["result_idx"] if "result_idx" in d_args else None
            return self.epoch_rule_instance_list[result_idx].obtain_metric(d_args)
        elif mode == ObtainMode.IdxWait:
            result_idx = d_args["result_idx"] if "result_idx" in d_args else None
            d_args_ = deepcopy(d_args)
            d_args_["result_idx"] = None
            d_args_["mode"] = ObtainMode.AllWait
            r = self.epoch_rule_instance_list[result_idx].obtain_metric(d_args_)
            self.single_epoch_module_metric_2da_list.append(r)
            return r
        elif mode == ObtainMode.AllWait:
            tmp = self.epoch_rule_instance_list[0]
            shape = (len(self.epoch_rule_instance_list), len(tmp.module_name_list), len(tmp.metric_suffix_list))
            epoch_module_metric_3da = np.zeros(shape)
            # 可能需要等intermediate_metric_rule的所有结果都出来才能计算
            for i in range(len(self.epoch_rule_instance_list)):
                epoch_module_metric_3da[i] = self.single_epoch_module_metric_2da_list[i]
            return epoch_module_metric_3da
        else:
            raise ValueError("mode {} not supported".format(mode))


class NotImplementMonitorRule(MonitorRuleBase):

    def record_metric(self, d_args):
        return

    def obtain_metric(self, d_args):
        return "nothing"


class ModePeriodMonitorRule(PeriodMonitorRuleBase):
    # train acc / val loss
    def __init__(self, d_args):
        super().__init__(d_args)
        self.key = d_args["key"]
        self.mode = getattr(RecordMode, d_args["mode_name"])

    def record_metric(self, d_args):
        mode = d_args["mode"]
        if mode == self.mode:
            return super().record_metric(d_args[self.key])

    def obtain_metric(self, d_args):
        return super().obtain_metric(d_args)


class ModeOnceMonitorRule(OnceMonitorRuleBase):
    # test acc
    def __init__(self, d_args):
        super().__init__(d_args)
        self.key = d_args["key"]
        self.mode = getattr(RecordMode, d_args["mode_name"])

    def record_metric(self, d_args):
        mode = d_args["mode"]
        if mode == self.mode:
            return super().record_metric(d_args[self.key])

    def obtain_metric(self, d_args):
        return super().obtain_metric(d_args)
