import logging

import torch
from torch import nn

from atdd_messenger import if_enable

logger = logging.getLogger(__name__)


class ATDDMonitor:
    def __init__(self, shared, report):
        logger.info("monitor hello!")
        self.shared = shared
        self.report = report
        self.complete_config_by_default()

        self.model_num = self.shared["model_num"]
        self.enable_dict = self.shared["enable_dict"]
        self.intermediate_default = self.report["intermediate_default"]
        self.final_default = self.report["final_default"]

        #
        self.total_layer_num = 0
        self.step_counter = 0

        #
        self.acc_list = []
        self.loss_list = []
        self.reward_list = []
        self.val_acc_list = []
        self.val_loss_list = []
        self.val_reward_list = []
        # self.poor_weight_list = []

        self.param_has_inf = False
        self.param_grad_zero_rate = None
        self.relu_module_name = None
        self.module_name_list = []
        self.module_name_flow_2dlist = []
        self.param_grad_abs_ave_list = []
        self.param_val_var_list = []  # 先 mean 再 var
        self.param_grad_var_list = []

        self.test_acc = None
        self.test_loss = None
        self.test_reward = None

        # DeepDiagnosis
        self.lr_cond = None
        self.weight_cond = None
        self.data_cond = None

        # self.improper_data = False
        # self.improper_weight_init = False
        # self.lr_condition = None
        # self.x_range = None
        # self.y_range = None

        # helper parameters:
        self.param_grad_nelement_total = 0
        self.param_grad_nzeroelement_total = 0

        self.param_grad_abs_ave_2dlist = []
        self.param_grad_var_2dlist = []
        self.param_val_var_2dlist = []

    def complete_config_by_default(self):
        pass

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

    def get_final_default_metric_value(self):
        if if_enable(["test"]) and "test" in self.intermediate_default:
            if if_enable(["acc"]) and "acc" in self.intermediate_default:
                return self.test_acc
            if if_enable(["reward"]) and "reward" in self.intermediate_default:
                return self.test_reward
            if if_enable(["loss"]) and "loss" in self.intermediate_default:
                return self.test_loss
        else:
            return self.get_intermediate_default_metric_value()

    def refresh_before_epoch_start(self):
        self.step_counter += 1

        # self.acc_list = []
        self.param_grad_abs_ave_list = []
        self.param_grad_var_list = []
        self.param_val_var_list = []
        self.param_grad_zero_rate = 0
        self.param_has_inf = False
        self.param_grad_nelement_total = 0
        self.param_grad_nzeroelement_total = 0
        self.param_grad_abs_ave_2dlist = []
        self.param_val_var_2dlist = []
        self.param_grad_var_2dlist = []
        for i in range(self.total_layer_num):
            self.param_grad_abs_ave_2dlist.append([])
            self.param_val_var_2dlist.append([])
            self.param_grad_var_2dlist.append([])
        # self.total_layer_num = 0

    def get_basic_v_result(self):
        d = {}
        if if_enable(["acc"]):
            d.update({"acc": self.acc_list[-1]})
        if if_enable(["loss"]):
            d.update({"loss": self.loss_list[-1]})
        if if_enable(["reward"]):
            d.update({"reward": self.reward_list[-1]})
        if if_enable(["acc", "val"]):
            d.update({"val_acc": self.val_acc_list[-1]})
        if if_enable(["loss", "val"]):
            d.update({"val_loss": self.val_loss_list[-1]})
        if if_enable(["reward", "val"]):
            d.update({"val_reward": self.val_reward_list[-1]})
        d.update({"step_counter": self.step_counter})
        return d

    def get_additional_v_result(self):
        d = {}
        d.update({"param_has_inf": self.param_has_inf})
        d.update({"param_grad_zero_rate": self.param_grad_zero_rate})
        d.update({"data_cond": self.data_cond})
        d.update({"weight_cond": self.weight_cond})
        d.update({"lr_cond": self.lr_cond})
        return d

    def get_basic_l_result(self):
        d = {}
        d.update({"acc_list": self.acc_list})
        d.update({"loss_list": self.loss_list})
        d.update({"reward_list": self.reward_list})
        d.update({"val_acc_list": self.val_acc_list})
        d.update({"val_loss_list": self.val_loss_list})
        d.update({"val_reward_list": self.val_reward_list})
        return d

    def get_result_4_assessor(self):
        # param_grad_abs_ave_list module_name_flow_2dlist module_name_list param_grad_zero_rate

        d = {}
        d.update({"param_grad_abs_ave_list": self.param_grad_abs_ave_list})
        d.update({"module_name_flow_2dlist": self.module_name_flow_2dlist})
        d.update({"module_name_list": self.module_name_list})
        return d

    def get_result_4_inspector(self):
        # param_grad_abs_ave_list param_has_inf param_grad_zero_rate acc_list val_acc_list
        # poor_weight_list loss_list val_loss_list
        # val_loss_list reward_list val_reward_list

        d = {}
        d.update({"param_has_inf": self.param_has_inf})
        d.update({"param_val_var_list": self.param_val_var_list})
        d.update({"param_grad_zero_rate": self.param_grad_zero_rate})
        d.update({"param_grad_abs_ave_list": self.param_grad_abs_ave_list})
        d.update({"module_name_flow_2dlist": self.module_name_flow_2dlist})
        d.update({"module_name_list": self.module_name_list})
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
        if if_enable(["acc","test"]):
            d.update({"test_acc": self.test_acc})
        if if_enable(["loss","test"]):
            d.update({"test_loss": self.test_loss})
        if if_enable(["reward","test"]):
            d.update({"test_reward": self.test_reward})
        return d

    def get_intermediate_dict(self):
        d = {"default": self.get_intermediate_default_metric_value()}
        d.update(self.get_basic_v_result())
        d.update(self.get_additional_v_result())
        d.update(self.get_result_4_inspector())
        d.update(self.get_result_4_assessor())
        d.update(self.get_basic_l_result())
        d.update(self.get_result_4_tuner())

        d.update({
            # for record:
            "param_grad_var_list": self.param_grad_var_list,  ####
            "param_val_var_list": self.param_val_var_list,  ####
        })
        return d

    def get_final_dict(self):
        d = {"default": self.get_final_default_metric_value()}
        d.update(self.get_basic_v_result())
        d.update(self.get_test_result())
        # d.update(self.get_result_4_inspector())
        # d.update(self.get_basic_l_result())
        d.update(self.get_result_4_tuner())

        return d

    def init_cond(self, opt, data_loader_list, lr):
        if if_enable(["opt"]):
            self.weight_cond = True
            for group in opt.param_groups:
                for param in group["params"]:
                    if torch.sum(param > 1) + torch.sum(param < -1) > 1:
                        self.weight_cond = False
        if if_enable(["data"]):
            self.data_cond = True
            x_range, y_range = [0, 0], [0, 0]
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
            if x_range[0] < -1 or x_range[1] > 1:
                self.data_cond = False
        if if_enable(["lr"]):
            if lr > 1e-3:
                self.lr_cond = True
            else:
                self.lr_cond = False

    def init_module_basic(self, model):
        if not if_enable(["model"]):
            return
        for (module_name, module) in model.named_modules():
            if type(module) in [nn.Conv2d, nn.Linear, nn.LSTM, nn.RNN]:
                self.module_name_list.append(module_name)
                for (param_name, param) in module.named_parameters():
                    if "weight" == param_name or ("weight" in param_name and "hh" in param_name):  # for lstm
                        self.total_layer_num += 1
                        self.param_grad_abs_ave_2dlist.append([])
                        self.param_val_var_2dlist.append([])
                        self.param_grad_var_2dlist.append([])
        self.relu_module_name = model.relu_module_name
        self.module_name_flow_2dlist = model.module_name_flow_2dlist

    def collect_in_training(self, model):
        if not if_enable(["model"]):
            return
        layer_index = 0
        # assume sequence idx->layer !!!!
        for (module_name, module) in model.named_modules():
            if type(module) in [nn.Conv2d, nn.Linear, nn.LSTM, nn.RNN]:
                for (param_name, param) in module.named_parameters():
                    if "weight" == param_name or ("weight" in param_name and "hh" in param_name):  # for lstm
                        self.param_grad_abs_ave_2dlist[layer_index].append(float(torch.mean(torch.abs(param.grad))))
                        # self.param_val_var_2dlist[layer_index].append(float(torch.var(torch.flatten(param))))
                        self.param_val_var_2dlist[layer_index].append(float(torch.mean(torch.flatten(param))))
                        self.param_grad_var_2dlist[layer_index].append(float(torch.var(torch.flatten(param.grad))))
                        layer_index += 1
                        self.param_grad_nelement_total += param.grad.nelement()

                        if True in [mid_name in module_name for mid_name in self.relu_module_name]:
                            self.param_grad_nzeroelement_total += torch.sum(param.grad == 0).item()
                        if True in torch.isinf(param) or True in torch.isinf(param.grad):
                            self.param_has_inf = True

    def collect_after_training(self, acc=None, loss=None, reward=None):
        if if_enable(["acc"]):
            self.acc_list.append(acc)
        if if_enable(["loss"]):
            self.loss_list.append(loss)
        if if_enable(["reward"]):
            self.reward_list.append(reward)

    def calculate_metrics_after_training(self):
        def get_ave(lst):
            return sum(lst) / len(lst)

        def get_var(lst):
            import numpy as np
            return float(np.var(lst))

        if if_enable(["model"]):
            for i in range(self.total_layer_num):
                self.param_grad_abs_ave_list.append(get_ave(self.param_grad_abs_ave_2dlist[i]))
                # self.param_val_var_list.append(get_ave(self.param_val_var_2dlist[i]))
                self.param_val_var_list.append(get_var(self.param_val_var_2dlist[i]))
                self.param_grad_var_list.append(get_ave(self.param_grad_var_2dlist[i]))
            self.param_grad_zero_rate = self.param_grad_nzeroelement_total / self.param_grad_nelement_total

    def collect_after_validating(self, acc=None, loss=None, reward=None):
        if if_enable(["val"]):
            if if_enable(["acc"]):
                self.val_acc_list.append(acc)
            if if_enable(["loss"]):
                self.val_loss_list.append(loss)
            if if_enable(["reward"]):
                self.val_reward_list.append(reward)

    def collect_after_testing(self, acc=None, loss=None, reward=None):
        if if_enable(["test"]):
            if if_enable(["acc"]):
                self.test_acc = acc
            if if_enable(["loss"]):
                self.test_loss = loss
            if if_enable(["reward"]):
                self.test_reward = reward
