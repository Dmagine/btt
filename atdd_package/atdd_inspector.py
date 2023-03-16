import logging

import nni
import numpy as np
import torch

from atdd_messenger import ATDDMessenger
from atdd_utils import diagnose_params
from atdd_utils import get_ave

logger = logging.getLogger(__name__)


class ATDDInspector:
    def __init__(self, shared, basic, diagnose):
        logger.info("inspector hello")
        self.shared = shared
        self.basic = basic
        self.diagnose = diagnose
        self.complete_config_by_default()

        self.model_num = self.shared["model_num"]
        self.max_epoch = self.shared["max_epoch"]
        self.enable_dict = self.shared["enable_dict"]
        self.start_step_float = self.basic["start_step_float"]
        self.rule_name_list = self.basic["rule_name_list"]
        self.dp = diagnose_params(**diagnose)

        self.start_step = int(self.start_step_float * self.max_epoch)
        self.window_size = int(self.dp.window_size_float * self.max_epoch)

        self.at_symptom = None
        self.dd_symptom = None
        self.wd_symptom = None

        self.last_acc = None
        self.acc_list = None
        self.loss_list = None
        self.reward_list = None
        self.val_acc_list = None
        self.val_loss_list = None
        self.val_reward_list = None
        self.param_has_inf = None
        self.param_grad_zero_rate = None
        self.param_val_var_list = None
        self.param_grad_abs_ave_list = None
        self.module_name_flow_2dlist = None
        self.module_name_list = None
        self.step_counter = None
        self.trial_id = None

        self.continuous_vg_count = 0  # alpha1
        self.continuous_eg_count = 0  # alpha2
        self.continuous_dr_count = 0  # alpha3

    def complete_config_by_default(self):
        pass

    def if_enable(self, lst):
        for name in lst:
            if name not in self.enable_dict:
                return False
            if self.enable_dict[name] is False:
                return False
        return True

    def get_default_symptom_dict(self):
        d = {"step_counter": self.step_counter,
             "continuous_vg_count": self.continuous_vg_count,
             "continuous_eg_count": self.continuous_eg_count,
             "continuous_dr_count": self.continuous_dr_count,
             "at_symptom": self.at_symptom,
             "dd_symptom": self.dd_symptom,
             "wd_symptom": self.wd_symptom,
             }
        return d

    def get_none_symptom_dict(self):
        d = {"step_counter": self.step_counter,
             "continuous_vg_count": None,
             "continuous_eg_count": None,
             "continuous_dr_count": None,
             "at_symptom": None,
             "dd_symptom": None,
             "wd_symptom": None,
             }
        return d

    def load_and_get_dict(self):
        msg = ATDDMessenger()
        d = msg.read_monitor_info()
        self.last_acc = d["acc_list"][-1]
        self.acc_list = d["acc_list"]
        self.loss_list = d["loss_list"]
        self.reward_list = d["reward_list"]
        self.val_acc_list = d["val_acc_list"]
        self.val_loss_list = d["val_loss_list"]
        self.val_reward_list = d["val_reward_list"]
        self.param_has_inf = d["param_has_inf"]
        self.param_grad_zero_rate = d["param_grad_zero_rate"]
        self.param_val_var_list = d["param_val_var_list"]
        self.param_grad_abs_ave_list = d["param_grad_abs_ave_list"]
        self.module_name_flow_2dlist = d["module_name_flow_2dlist"]
        self.module_name_list = d["module_name_list"]
        self.step_counter = d["step_counter"]

        if self.step_counter < self.start_step:
            logger.info(" ".join(["step_counter:", str(self.step_counter), "lt", "start_step", str(self.start_step)]))
            return self.get_none_symptom_dict()
        # if self.top_performance():
        #     return self.get_none_symptom_dict()

        self.judge_symptom()
        d = self.get_default_symptom_dict()
        logger.info(" ".join(["default_symptom_dict:", str(d)]))
        return d

    def judge_at_symptom(self):
        if self.if_vg():
            self.at_symptom = "VG"
            return True
        if self.if_eg():
            self.at_symptom = "EG"
            return True
        if self.if_dr():
            self.at_symptom = "DR"
            return True
        if self.if_ol():
            self.at_symptom = "OL"
            return True
        if self.if_sc():
            self.at_symptom = "SC"
            return True
        return False

    def judge_dd_symptom(self):
        # dd
        if self.if_dd_et():
            self.dd_symptom = "ExplodingTensor"
            return True
        if self.if_dd_uw():
            self.dd_symptom = "UnchangedWeight"  # or PoorWeight ? DD use no param_grad_var_list
            return True

        # if self.dd_sa_judge():  # unused
        #     self.dd_symptom = "SaturatedActivation"
        #     return True
        # if self.dd_dn_judge():  # unused
        #     self.dd_symptom = "DeadNode"
        #     return True
        # if self.dd_or_judge():  # unused
        #     self.dd_symptom = "Outofange"
        #     return True

        if self.if_dd_lnd():
            self.dd_symptom = "LossNotDecreasing"
            return True
        if self.if_dd_ani():
            self.dd_symptom = "AccuracyNotIncreasing"
            return True
        if self.if_dd_vg():
            self.dd_symptom = "VanishingGradient"
            return True
        return False

    def judge_wd_symptom(self):
        if self.if_wd_symptom("acc"):
            self.wd_symptom = "AccuracyWindowNotGood"
            return True
        if self.if_wd_symptom("loss"):
            self.wd_symptom = "LossWindowNotGood"
            return True
        if self.if_wd_symptom("reward"):
            self.wd_symptom = "RewardWindowNotGood"
            return True
        return False

    def top_performance(self):
        if self.acc_list is not None:
            if self.last_acc >= np.percentile(self.acc_list, 100):  # [1,2,3]100 ->3 !!!!!!!
                logger.info(" ".join(["top acc performance:", nni.get_trial_id(), str(self.last_acc)]))
                return True
        if self.loss_list is not None and len(self.loss_list) > 0:
            if self.loss_list[-1] <= np.percentile(self.loss_list, 0):
                logger.info(" ".join(["top loss performance:", nni.get_trial_id(), str(self.loss_list[-1])]))
                return True
        return False

    def judge_symptom(self):
        symptom_flag = False
        if "at" in self.rule_name_list:
            symptom_flag |= self.judge_at_symptom()
        if "dd" in self.rule_name_list:
            symptom_flag |= self.judge_dd_symptom()
        if "wd" in self.rule_name_list:
            symptom_flag |= self.judge_wd_symptom()

        return symptom_flag

    def _if_wd_symptom(self, li, expect: str):
        if len(li) >= self.window_size: ###
            s = self.window_size
            mean_now = get_ave(li[-s:])
            var_now = float(torch.var(torch.tensor(li[-s:])))
            if len(li) >= self.window_size * 2:
                mean_pre = get_ave(li[-s * 2:-s])
                var_pre = float(torch.var(torch.tensor(li[-s * 2:-s])))
            else:
                # 改 -s  s
                mean_pre = get_ave(li[:s])
                var_pre = float(torch.var(torch.tensor(li[:s])))

            if var_now > var_pre:
                return True
            if expect == "inc" and mean_now < mean_pre:
                return True
            if expect == "dec" and mean_now > mean_pre:
                return True
        return False

    def if_wd_symptom(self, s):
        if s == "acc":
            return self._if_wd_symptom(self.acc_list, "inc") and self._if_wd_symptom(self.val_acc_list, "inc")
        elif s == "loss":
            return self._if_wd_symptom(self.loss_list, "dec") and self._if_wd_symptom(self.val_loss_list, "dec")
        elif s == "reward":
            return self._if_wd_symptom(self.reward_list, "inc") and self._if_wd_symptom(self.val_reward_list, "inc")

    def if_dd_et(self):
        if not self.if_enable(["model"]):
            return False
        if self.param_has_inf:
            return True
        return False

    def if_dd_uw(self):
        if not self.if_enable(["model"]):
            return False
        if self.step_counter >= self.window_size:
            poor_weight_list = []
            for i in range(len(self.param_val_var_list)):
                if self.param_val_var_list[i] < self.dp.dd_min_threshold \
                        or self.param_val_var_list[i] > self.dp.dd_max_threshold:
                    poor_weight_list.append(True)
                else:
                    poor_weight_list.append(False)
            if True in poor_weight_list:  # 任何一层有问题就停止
                return True
        return False

    def _if_dd_lnd(self, loss_list):
        if not self.if_enable(["loss"]):
            return False
        if self.step_counter >= self.window_size: ###
            if loss_list[-1] >= get_ave(loss_list[-self.window_size:]): ###
                return True
        return False

    def if_dd_lnd(self):
        return self._if_dd_lnd(self.loss_list) and self._if_dd_lnd(self.val_loss_list)

    def if_dd_ani(self):
        if not self.if_enable(["acc"]):
            return False
        if self.step_counter >= self.window_size:
            if self.last_acc <= get_ave(self.acc_list[-self.window_size:]): ###
                return True
        return False

    def if_dd_vg(self):
        if not self.if_enable(["model"]):
            return False
        for item in self.param_grad_abs_ave_list:
            if item < self.dp.dd_threshold_VG:
                return True
        return False

    def if_vg(self):
        if not self.if_enable(["acc", "model"]):
            return False

        symptom_flag = False
        if 0 in self.param_grad_abs_ave_list:
            symptom_flag = True
        else:
            for module_name_flow_list in self.module_name_flow_2dlist:
                module_idx_flow_list = [self.module_name_list.index(name) for name in module_name_flow_list]
                if self.last_acc <= self.dp.theta and \
                        self.param_grad_abs_ave_list[module_idx_flow_list[0]] <= self.dp.beta2:
                    for i in range(len(module_idx_flow_list) - 1):
                        idx_1 = module_idx_flow_list[i]
                        idx_2 = module_idx_flow_list[i + 1]
                        if self.param_grad_abs_ave_list[idx_1] / self.param_grad_abs_ave_list[idx_2] <= self.dp.beta1:
                            symptom_flag = True
                            break
                    if symptom_flag:
                        break
        self.continuous_vg_count = self.continuous_vg_count + 1 if symptom_flag else 0
        if self.continuous_vg_count > self.dp.alpha1:
            return True
        return False

    def if_eg(self):
        if not self.if_enable(["acc", "model"]):
            return False
        symptom_flag = False
        if self.param_has_inf:
            symptom_flag = True  ####
        else:
            for module_name_flow_list in self.module_name_flow_2dlist:
                module_idx_flow_list = [self.module_name_list.index(name) for name in module_name_flow_list]
                if self.last_acc <= self.dp.theta:
                    for i in range(len(module_idx_flow_list) - 1):
                        idx_1 = module_idx_flow_list[i]
                        idx_2 = module_idx_flow_list[i + 1]
                        if self.param_grad_abs_ave_list[idx_1] / self.param_grad_abs_ave_list[idx_2] >= self.dp.beta3:
                            symptom_flag = True
                            break
                    if symptom_flag:
                        break
        self.continuous_eg_count = self.continuous_eg_count + 1 if symptom_flag else 0
        if self.continuous_eg_count > self.dp.alpha2:
            return True
        return False

    def if_dr(self):
        if not self.if_enable(["acc", "model"]):
            return False
        symptom_flag = False
        if self.last_acc <= self.dp.beta3 and self.param_grad_zero_rate >= self.dp.gamma:
            symptom_flag = True
        self.continuous_dr_count = self.continuous_dr_count + 1 if symptom_flag else 0
        if self.continuous_dr_count > self.dp.alpha3:
            return True
        return False

    def get_ol_metric(self, acc_list):
        count = 0
        maximum_list = []
        minimum_list = []
        for i in range(len(acc_list)):
            if i == 0 or i == len(acc_list) - 1:
                continue
            if acc_list[i] - acc_list[i - 1] >= 0 and acc_list[i] - acc_list[i + 1] >= 0:
                maximum_list.append(acc_list[i])
            if acc_list[i] - acc_list[i - 1] <= 0 and acc_list[i] - acc_list[i + 1] <= 0:
                minimum_list.append(acc_list[i])
        for i in range(min(len(maximum_list), len(minimum_list))):
            if maximum_list[i] - minimum_list[i] >= self.dp.zeta:
                count += 1
        return count / len(acc_list)

    def if_ol(self):
        if not self.if_enable(["acc"]):
            return False
        ol_rate = self.get_ol_metric(self.acc_list)
        if ol_rate >= self.dp.eta:
            return True
        return False

    def if_sc(self):
        return self._if_sc(self.acc_list) and self._if_sc(self.val_acc_list)

    def _if_sc(self, acc_list):
        # 任意的 i 都要满足 才触发！
        if not self.if_enable(["acc"]):
            return False
        if len(acc_list) > 1:
            for i in range(1, len(acc_list)):  # !!!!! 0- -1
                if acc_list[i] - acc_list[i - 1] > self.dp.delta:
                    return False
            # logger.warning("%s: %s\n" % (self.tmp_trial_job_id, "problem SC"))
            return True
        return False
