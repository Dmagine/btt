import logging

import numpy as np
import scipy.stats as stats
import torch
from torch import nn

from atdd_messenger import if_enable

logger = logging.getLogger(__name__)


class ATDDMonitor:
    def __init__(self, shared, report):
        logger.info("monitor hello!")
        self.shared = shared
        self.report = report
        # self.calc = calc
        self.complete_config_by_default()

        self.model_num = self.shared["model_num"]
        self.enable_dict = self.shared["enable_dict"]
        self.max_epoch = self.shared["max_epoch"]
        self.quick_calc = self.shared["quick_calc"]
        self.intermediate_default = self.report["intermediate_default"]
        self.final_default = self.report["final_default"]

        self.epoch_idx = None
        self.batch_idx = None

        #
        self.acc_list = None
        self.loss_list = None
        self.reward_list = None
        self.val_acc_list = None
        self.val_loss_list = None
        self.val_reward_list = None
        self.test_acc = None
        self.test_loss = None
        self.test_reward = None

        self.has_nan_inf_list = None
        self.epoch_has_nan_inf = None

        self.module_nele_list = None
        self.module_name_flow_matrix = None
        self.relu_pre_module_name_list = None
        # self.module_name_list = None
        # self.relu_pre_module_name = None

        # 跨网络层统筹的统计值（将不同网络层一视同仁取均值） 一个epoch对应一个值 ##############

        # 原始：一个epoch的一个batch的  |  一个module的(一个对象的)一个统计值
        # module内处理(均值/中值/末值)：cube -> matrix
        # batch处理(均值/中值/末值)：matrix -> list
        # module间处理(均值/中值/末值)：list -> value

        # DeepDiagnosis
        self.lr_cond = None
        self.weight_cond = None
        self.data_cond = None

        self.num_train_batch = None

        self.module_name_list = None
        self.metric_name_list = None
        self.metric_prefix_list = ["weight", "weight_abs", "weight_grad", "weight_grad_abs"]
        self.metric_suffix_list = ["avg", "var", "mid", "max", "min", "upper", "lower", "skew", "kurt", "rate0"]

        self.epoch_module_metric_3da = None  # dim1:epoch_idx, dim2:module_idx, dim3:metric_idx
        self.batch_module_metric_3da = None  # dim1:batch_idx, dim2:module_idx, dim3:metric_idx

        self.epoch_module_weight_grad_abs_avg_2da = None  # dim1:epoch_idx, dim2:module_idx
        self.epoch_module_weight_grad_rate0_2da = None  # dim1:batch_idx, dim2:module_idx

        self.batch_module_weight_grad_abs_avg_2da = None  # dim1:epoch_idx, dim2:module_idx
        self.batch_module_weight_grad_rate0_2da = None  # dim1:batch_idx, dim2:module_idx

        # [nn.Conv2d, nn.Linear, nn.Conv1d, nn.LSTMCell]
        self.support_module_type_list = [nn.Conv2d, nn.Linear, nn.Conv1d, nn.LSTM]

    def complete_config_by_default(self):
        pass

    def refresh_before_epoch_start(self):
        self.epoch_idx = self.epoch_idx + 1 if self.epoch_idx is not None else 0
        self.batch_idx = 0
        self.epoch_has_nan_inf = False

    def get_intermediate_default_metric_value(self):
        if if_enable(["val"]) and "val" in self.intermediate_default:  # e.g. "val_acc"
            if if_enable(["acc"]) and "acc" in self.intermediate_default:
                return self.val_acc_list[-1]
            if if_enable(["reward"]) and "reward" in self.intermediate_default:
                return self.val_reward_list[-1]
            if if_enable(["loss"]) and "loss" in self.intermediate_default:
                return self.val_loss_list[-1]
        else:
            if if_enable(["acc"]) and "acc" in self.intermediate_default:
                return self.acc_list[-1]
            if if_enable(["reward"]) and "reward" in self.intermediate_default:
                return self.reward_list[-1]
            if if_enable(["loss"]) and "loss" in self.intermediate_default:
                return self.loss_list[-1]
        # not fit for self.intermediate_default
        if if_enable(["val"]):  # e.g. "val_acc"
            if if_enable(["acc"]):
                return self.val_acc_list[-1]
            if if_enable(["reward"]):
                return self.val_reward_list[-1]
            if if_enable(["loss"]):
                return self.val_loss_list[-1]
        else:
            if if_enable(["acc"]):
                return self.acc_list[-1]
            if if_enable(["reward"]):
                return self.reward_list[-1]
            if if_enable(["loss"]):
                return self.loss_list[-1]

    def get_final_default_metric_value(self):
        if if_enable(["test"]) and "test" in self.intermediate_default:
            if if_enable(["acc"]) and "acc" in self.intermediate_default:
                return self.test_acc
            if if_enable(["reward"]) and "reward" in self.intermediate_default:
                return self.test_reward
            if if_enable(["loss"]) and "loss" in self.intermediate_default:
                return self.test_loss
            # not fit for self.intermediate_default
            if if_enable(["acc"]):
                return self.test_acc
            if if_enable(["reward"]):
                return self.test_reward
            if if_enable(["loss"]):
                return self.test_loss
        else:
            return self.get_intermediate_default_metric_value()

    def get_basic_v_result(self):
        d = {}
        if if_enable(["acc"]) and self.acc_list is not None:
            d.update({"acc": self.acc_list[-1]})
        if if_enable(["loss"]) and self.loss_list is not None:
            d.update({"loss": self.loss_list[-1]})
        if if_enable(["reward"]) and self.reward_list is not None:
            d.update({"reward": self.reward_list[-1]})
        if if_enable(["acc", "val"]) and self.val_acc_list is not None:
            d.update({"val_acc": self.val_acc_list[-1]})
        if if_enable(["loss", "val"]) and self.val_loss_list is not None:
            d.update({"val_loss": self.val_loss_list[-1]})
        if if_enable(["reward", "val"]) and self.val_reward_list is not None:
            d.update({"val_reward": self.val_reward_list[-1]})
        d.update({"epoch_idx": self.epoch_idx})
        return d

    def get_statistic_2da_result(self):
        d = {}
        if self.quick_calc:
            d.update({"weight_grad_abs_avg_1da": self.epoch_module_weight_grad_abs_avg_2da[self.epoch_idx]})
            d.update({"weight_grad_rate0_1da": self.epoch_module_weight_grad_rate0_2da[self.epoch_idx]})
        else:
            d.update({"module_metric_2da": self.epoch_module_metric_3da[self.epoch_idx]})
        return d

    def get_basic_l_result(self):
        d = {}
        if if_enable(["acc"]) and self.acc_list is not None:
            d.update({"acc_list": self.acc_list})
        if if_enable(["loss"]) and self.loss_list is not None:
            d.update({"loss_list": self.loss_list})
        if if_enable(["reward"]) and self.reward_list is not None:
            d.update({"reward_list": self.reward_list})
        if if_enable(["acc", "val"]) and self.val_acc_list is not None:
            d.update({"val_acc_list": self.val_acc_list})
        if if_enable(["loss", "val"]) and self.val_loss_list is not None:
            d.update({"val_loss_list": self.val_loss_list})
        if if_enable(["reward", "val"]) and self.val_reward_list is not None:
            d.update({"val_reward_list": self.val_reward_list})
        return d

    def get_result_4_inspector_assessor(self):
        d = {}
        d.update({"has_nan_inf_list": self.has_nan_inf_list})
        # d.update({"module_name_flow_matrix": self.module_name_flow_matrix})
        # d.update({"relu_pre_module_name_list": self.relu_pre_module_name_list})
        d.update({"module_name_list": self.module_name_list})
        d.update({"module_nele_list": self.module_nele_list})

        #
        d.update({"epoch_idx": self.epoch_idx})
        if self.quick_calc:
            d.update({"weight_grad_abs_avg_1da": self.epoch_module_weight_grad_abs_avg_2da[self.epoch_idx]})
            d.update({"weight_grad_rate0_1da": self.epoch_module_weight_grad_rate0_2da[self.epoch_idx]})
        else:
            d.update({"module_metric_2da": self.epoch_module_metric_3da[self.epoch_idx]})
            d.update({"metric_prefix_list": self.metric_prefix_list})
            d.update({"metric_suffix_list": self.metric_suffix_list})
        return d

    def get_result_4_tuner(self):
        d = {}
        d.update({"data_cond": self.data_cond})
        d.update({"weight_cond": self.weight_cond})
        d.update({"lr_cond": self.lr_cond})
        # d.update({"trial_id": nni.get_trial_id()})
        return d

    def get_test_result(self):
        d = {}
        if if_enable(["acc", "test"]):
            d.update({"test_acc": self.test_acc})
        if if_enable(["loss", "test"]):
            d.update({"test_loss": self.test_loss})
        if if_enable(["reward", "test"]):
            d.update({"test_reward": self.test_reward})
        return d

    def get_intermediate_dict(self):

        d = {"default": self.get_intermediate_default_metric_value()}
        d.update(self.get_basic_v_result())
        d.update(self.get_result_4_inspector_assessor())
        d.update(self.get_basic_l_result())
        d.update(self.get_statistic_2da_result())
        return d

    def get_final_dict(self):
        tmp = self.epoch_idx + 1
        if self.quick_calc:
            self.epoch_module_weight_grad_abs_avg_2da = self.epoch_module_weight_grad_abs_avg_2da[:tmp, :]
            self.epoch_module_weight_grad_rate0_2da = self.epoch_module_weight_grad_rate0_2da[:tmp, :]
        else:
            self.epoch_module_metric_3da = self.epoch_module_metric_3da[:tmp, :, :]
        d = {"default": self.get_final_default_metric_value()}
        d.update(self.get_basic_v_result())
        d.update(self.get_test_result())
        d.update(self.get_statistic_2da_result())
        return d

    def init_cond(self, opt, data_loader_list, lr):
        if if_enable(["opt"]):
            self.weight_cond = True
            for group in opt.param_groups:
                for param in group["params"]:
                    if torch.sum(param > 1) + torch.sum(param < -1) > 1:
                        self.weight_cond = False
        if if_enable(["data"]):
            # 提速 一般一个dataloader的几个batch就能判断出了
            self.data_cond = True
            x_range, y_range = [0, 0], [0, 0]
            count, n = 0, 5
            for dataloader in data_loader_list:
                for x, y in dataloader:
                    if x_range is None:
                        x_range, y_range = [0, 0], [0, 0]
                        x_range[0] = float(torch.min(x))
                        x_range[1] = float(torch.max(x))
                        y_range[0] = float(torch.min(y))
                        y_range[1] = float(torch.max(y))
                    else:
                        x_range[0] = min(x_range[0], float(torch.min(x)))
                        x_range[1] = max(x_range[1], float(torch.max(x)))
                        y_range[0] = min(y_range[0], float(torch.min(y)))
                        y_range[1] = max(y_range[1], float(torch.max(y)))
                    count += 1
                    if count > n:
                        break
                    if x_range[0] < -1 or x_range[1] > 1:
                        self.data_cond = False
                        break
                break

        if if_enable(["lr"]):
            self.lr_cond = True if lr > 1e-3 else False

    def init_basic(self, model, train_dataloader):
        if not if_enable(["model"]):
            return
        self.has_nan_inf_list = []

        self.num_train_batch = len(train_dataloader)
        self.module_name_list = []
        self.module_nele_list = []
        for (module_name, module) in model.named_modules():
            if type(module) not in self.support_module_type_list:
                continue
            for (param_name, param) in module.named_parameters():
                if "weight" not in param_name:
                    continue
                self.module_name_list.append(module_name)
                self.module_nele_list.append(param.nelement())
                break  # 只统计一次
        self.metric_name_list = []
        for prefix in self.metric_prefix_list:
            for suffix in self.metric_suffix_list:
                self.metric_name_list.append("_".join([prefix, suffix]))

        if self.quick_calc is True:
            self.batch_module_weight_grad_abs_avg_2da = np.zeros((self.num_train_batch, len(self.module_name_list)))
            self.batch_module_weight_grad_rate0_2da = np.zeros((self.num_train_batch, len(self.module_name_list)))
            self.epoch_module_weight_grad_abs_avg_2da = np.zeros((self.max_epoch, len(self.module_name_list)))
            self.epoch_module_weight_grad_rate0_2da = np.zeros((self.max_epoch, len(self.module_name_list)))
        else:
            self.batch_module_metric_3da = \
                np.zeros((self.num_train_batch, len(self.module_name_list), len(self.metric_name_list)))
            self.epoch_module_metric_3da = np.zeros(
                (self.max_epoch, len(self.module_name_list), len(self.metric_name_list)))

        logger.debug(" ".join([" ", "module_name_list:", str(self.module_name_list)]))
        logger.debug(" ".join([" ", "metric_name_list:", str(self.metric_name_list)]))

        # 一般来说 展开后的一个module 对应一个weight
        # print(module_name,module_name.split('.'),param_name)
        # conv1.0 ['conv1', '0'] weight
        # conv1.0 ['conv1', '0'] bias
        # blk1.conv1 ['blk1', 'conv1'] weight
        # blk1.conv1 ['blk1', 'conv1'] bias
        # blk1.conv2 ['blk1', 'conv2'] weight
        # blk1.conv2 ['blk1', 'conv2'] bias
        # blk1.extra.0 ['blk1', 'extra', '0'] weight
        # blk1.extra.0 ['blk1', 'extra', '0'] bias
        # blk2.conv1 ['blk2', 'conv1'] weight
        # blk2.conv1 ['blk2', 'conv1'] bias
        # blk2.conv2 ['blk2', 'conv2'] weight
        # blk2.conv2 ['blk2', 'conv2'] bias

        self.module_name_flow_matrix = model.module_name_flow_matrix \
            if hasattr(model, "module_name_flow_matrix") else [self.module_name_list]
        self.relu_pre_module_name_list = model.relu_pre_module_name_list \
            if hasattr(model, "relu_pre_module_name_list") else self.module_name_list

    def clean_tensor(self, x):
        # self.has_inf_list
        x = x[~ torch.isnan(x)]
        x_finite = x[~ torch.isinf(x)]
        if x_finite.nelement() == 0:
            return x
        x_max = torch.max(x_finite)
        x_min = torch.min(x_finite)
        x[torch.inf == x] = x_max
        x[-torch.inf == x] = x_min
        return x

    def collect_in_training(self, model):
        if not if_enable(["model"]):
            return

        module_idx = 0
        if self.quick_calc:
            pass
        else:
            single_batch_module_metric_2da = np.zeros((len(self.module_name_list), len(self.metric_name_list)))
        for (module_name, module) in model.named_modules():
            if type(module) not in self.support_module_type_list:
                continue
            for (param_name, param) in module.named_parameters():
                if "weight" not in param_name:
                    continue

                # weight_grad_array = self.clean_tensor(param.grad.flatten().detach().cpu()).numpy()
                # weight_grad_abs_array = np.abs(weight_grad_array)
                weight_grad_array = (param.grad.flatten().detach().cpu()).numpy()
                weight_grad_abs_array = np.abs(weight_grad_array)

                if self.quick_calc:
                    tmp = weight_grad_abs_array
                    self.batch_module_weight_grad_abs_avg_2da[self.batch_idx][module_idx] = np.mean(tmp)
                    tmp = weight_grad_array
                    self.batch_module_weight_grad_rate0_2da[self.batch_idx][module_idx] = np.sum(tmp == 0) / tmp.size
                    self.epoch_has_nan_inf &= np.any(np.isinf(weight_grad_array))
                else:
                    weight_array = (param.flatten().detach().cpu()).numpy()
                    weight_abs_array = np.abs(weight_array)
                    metric_idx = 0
                    for prefix in self.metric_prefix_list:
                        tmp = None
                        # ["weight", "weight_abs", "weight_grad", "weight_grad_abs"]
                        if prefix == "weight":
                            tmp = weight_array
                        elif prefix == "weight_abs":
                            tmp = weight_abs_array
                        elif prefix == "weight_grad":
                            tmp = weight_grad_array
                        elif prefix == "weight_grad_abs":
                            tmp = weight_grad_abs_array
                        # ["avg", "var", "mid", "max", "min", "upper", "lower", "skew", "kurt", "rate0"]
                        for suffix in self.metric_suffix_list:
                            metric_name = prefix + "_" + suffix
                            if metric_name not in self.metric_name_list:
                                single_batch_module_metric_2da[module_idx][metric_idx] = 0  # fast
                                metric_idx += 1
                                continue
                            if tmp.size == 0:
                                single_batch_module_metric_2da[module_idx][metric_idx] = np.nan
                                self.epoch_has_nan_inf = True
                                metric_idx += 1
                                continue
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
                            metric_idx += 1
                    self.batch_module_metric_3da[self.batch_idx] = single_batch_module_metric_2da
                    self.epoch_has_nan_inf &= np.any(np.isinf(weight_array)) or np.any(np.isinf(weight_grad_array))

                module_idx += 1
                break  # only calculate the first weight
        self.batch_idx += 1

    def collect_after_training(self, acc=None, loss=None, reward=None):
        if if_enable(["acc"]) and acc is not None:
            self.acc_list = [] if self.acc_list is None else self.acc_list
            self.acc_list.append(acc)
        if if_enable(["loss"]) and loss is not None:
            self.loss_list = [] if self.loss_list is None else self.loss_list
            self.loss_list.append(loss)
        if if_enable(["reward"]) and reward is not None:
            self.reward_list = [] if self.reward_list is None else self.reward_list
            self.reward_list.append(reward)

    def calculate_after_training(self):
        if if_enable(["model"]):
            if self.quick_calc:
                tmp = np.mean(self.batch_module_weight_grad_abs_avg_2da, axis=0)
                self.epoch_module_weight_grad_abs_avg_2da[self.epoch_idx] = tmp
                tmp = np.mean(self.batch_module_weight_grad_rate0_2da, axis=0)
                self.epoch_module_weight_grad_rate0_2da[self.epoch_idx] = tmp
            else:
                self.epoch_module_metric_3da[self.epoch_idx] = np.mean(self.batch_module_metric_3da, axis=0)

        self.has_nan_inf_list.append(self.epoch_has_nan_inf)

    def collect_after_validating(self, acc=None, loss=None, reward=None):
        if if_enable(["val"]):
            if if_enable(["acc"]) and acc is not None:
                self.val_acc_list = [] if self.val_acc_list is None else self.val_acc_list
                self.val_acc_list.append(acc)
            if if_enable(["loss"]) and loss is not None:
                self.val_loss_list = [] if self.val_loss_list is None else self.val_loss_list
                self.val_loss_list.append(loss)
            if if_enable(["reward"]) and reward is not None:
                self.val_reward_list = [] if self.val_reward_list is None else self.val_reward_list
                self.val_reward_list.append(reward)

    def collect_after_testing(self, acc=None, loss=None, reward=None):
        if if_enable(["test"]):
            if if_enable(["acc"]) and acc is not None:
                self.test_acc = acc
            if if_enable(["loss"]) and loss is not None:
                self.test_loss = loss
            if if_enable(["reward"]) and reward is not None:
                self.test_reward = reward


def get_methods(self):
    lst1 = list(filter(lambda m: not m.startswith("__") and callable(getattr(self, m)), dir(self)))
    print("关键方法:", lst1)
    lst2 = list(filter(lambda m: not m.startswith("__"), dir(self)))
    print("关键变量", lst2)


if __name__ == '__main__':
    get_methods(ATDDMonitor)
