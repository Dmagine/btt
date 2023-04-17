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
        self.quick_calc = self.shared["quick_calc"]

        self.start_step_float = self.basic["start_step_float"]
        self.rule_name_list = self.basic["rule_name_list"]
        self.continuous_trigger_count = self.basic["continuous_trigger_count"]
        self.dp = diagnose_params(**diagnose)

        self.start_step = int(self.start_step_float * self.max_epoch)
        self.window_size = max(round(self.dp.window_size_float * self.max_epoch), self.dp.window_size_min)

        self.at_symptom = None
        self.dd_symptom = None
        self.wd_symptom = None

        self.last_acc = None
        self.acc_list = None
        self.loss_list = None
        self.reward_list = None
        self.reward_list = None
        self.val_acc_list = None
        self.val_loss_list = None
        self.val_reward_list = None
        self.epoch_has_nan_inf = None
        self.weight_grad_rate0 = None
        self.param_val_var_list = None
        self.weight_grad_abs_avg_1da = None
        self.module_name_flow_matrix = None
        self.module_name_list = None
        self.epoch_idx = None
        self.trial_id = None

        self.cnt_vg_count = 0
        self.cnt_eg_count = 0
        self.cnt_dr_count = 0
        self.cnt_sc_count = 0
        self.cnt_ol_count = 0
        self.cnt_ani_count = 0
        self.cnt_lnd_count = 0
        self.cnt_vg2_count = 0
        self.cnt_et_count = 0

        self.module_metric_2da = None
        self.metric_prefix_list = None
        self.metric_suffix_list = None

        self.relu_pre_module_name_list = None  # not used
        self.module_nele_list = None

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
        d = {
            "cnt_vg_count": self.cnt_vg_count,
            "cnt_eg_count": self.cnt_eg_count,
            "cnt_dr_count": self.cnt_dr_count,
            "cnt_sc_count": self.cnt_sc_count,
            "cnt_ol_count": self.cnt_ol_count,
            "cnt_ani_count": self.cnt_ani_count,
            "cnt_lnd_count": self.cnt_lnd_count,
            "cnt_vg2_count": self.cnt_vg2_count,
            "cnt_et_count": self.cnt_et_count,
            "at_symptom": self.at_symptom,
            "dd_symptom": self.dd_symptom,
            "wd_symptom": self.wd_symptom,
        }
        return d

    def get_none_symptom_dict(self):
        d = {
            "cnt_vg_count": self.cnt_vg_count,
            "cnt_eg_count": self.cnt_eg_count,
            "cnt_dr_count": self.cnt_dr_count,
            "cnt_sc_count": self.cnt_sc_count,
            "cnt_ol_count": self.cnt_ol_count,
            "cnt_ani_count": self.cnt_ani_count,
            "cnt_lnd_count": self.cnt_lnd_count,
            "cnt_vg2_count": self.cnt_vg2_count,
            "cnt_et_count": self.cnt_et_count,
            "at_symptom": None,
            "dd_symptom": None,
            "wd_symptom": None,
        }
        return d

    def get_metric_array(self, p, s):
        idx = self.metric_prefix_list.index(p) * len(self.metric_suffix_list) + self.metric_suffix_list.index(s)
        return self.module_metric_2da[:, idx].flatten()

    def load_and_get_dict(self):

        msg = ATDDMessenger()
        d = msg.read_monitor_info()
        self.last_acc = d["acc_list"][-1] if "acc_list" in d else None
        self.acc_list = d["acc_list"] if "acc_list" in d else None
        self.loss_list = d["loss_list"] if "loss_list" in d else None
        self.reward_list = d["reward_list"] if "reward_list" in d else None
        self.val_acc_list = d["val_acc_list"] if "val_acc_list" in d else None
        self.val_loss_list = d["val_loss_list"] if "val_loss_list" in d else None
        self.val_reward_list = d["val_reward_list"] if "val_reward_list" in d else None

        # self.param_has_inf = d["param_has_inf"]
        # self.param_grad_zero_rate = d["param_grad_zero_rate"]
        # self.param_val_var_list = d["param_val_var_list"]  # 不好说明。。。。。uw废弃
        # self.param_grad_abs_ave_list = d["param_grad_abs_ave_list"]
        # self.module_name_flow_2dlist = d["module_name_flow_2dlist"]
        # self.module_name_list = d["module_name_list"]
        # self.step_counter = d["step_counter"]

        #         d = {}
        #         d.update({"has_inf_list": self.has_inf_list})
        #         d.update({"module_name_flow_matrix": self.module_name_flow_matrix})
        #         d.update({"relu_pre_module_name_list": self.relu_pre_module_name_list})
        #         d.update({"module_name_list": self.module_name_list})
        #         d.update({"module_nelement_list": self.module_nelement_list})
        #
        #         # nele
        #         d.update({"metric_prefix_list": self.metric_prefix_list})
        #         d.update({"metric_suffix_list": self.metric_suffix_list})
        #         d.update({"module_metric_2da": self.epoch_module_metric_3da[self.epoch_idx]})

        self.module_nele_list = d["module_nele_list"]
        self.relu_pre_module_name_list = d["relu_pre_module_name_list"] \
            if "relu_pre_module_name_list" in d else d["module_name_list"]
        self.epoch_has_nan_inf = d["has_nan_inf_list"][-1]

        if self.quick_calc:
            self.weight_grad_abs_avg_1da = d["weight_grad_abs_avg_1da"]
            self.weight_grad_rate0 = np.average(d["weight_grad_rate0_1da"], 0, self.module_nele_list)
        else:
            self.module_metric_2da = d["module_metric_2da"]
            self.metric_prefix_list = d["metric_prefix_list"]
            self.metric_suffix_list = d["metric_suffix_list"]
            self.weight_grad_rate0 = np.average(self.get_metric_array("weight_grad", "rate0"), 0, self.module_nele_list)
            self.weight_grad_abs_avg_1da = self.get_metric_array("weight_grad_abs", "avg")

        self.epoch_idx = d["epoch_idx"]
        self.module_name_flow_matrix = d["module_name_flow_matrix"] \
            if "module_name_flow_matrix" in d else [d["module_name_list"]]
        self.module_name_list = d["module_name_list"]

        if self.epoch_idx < self.start_step:
            logger.info(" ".join(["step_counter:", str(self.epoch_idx), "lt", "start_step", str(self.start_step)]))
            return self.get_none_symptom_dict()
        # if self.if_top_performance():
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
        ############################ 。。。。。 # 细看 rfi4zmp0 还是合理（10*-5）！但是 unpb3fiz 受其影响（10*-10）
        # if self.if_dd_uw():
        #     self.dd_symptom = "UnchangedWeight"  # or PoorWeight ? DD use no param_grad_var_list ...
        #     return True

        # if self.dd_sa_judge():  # unused
        #     self.dd_symptom = "SaturatedActivation" # register hook
        #     return True
        # if self.dd_dn_judge():  # unused
        #     self.dd_symptom = "DeadNode"
        #     return True
        # if self.dd_or_judge():  # unused
        #     self.dd_symptom = "Outofrange" # data can't change
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
        # if "wd" in self.rule_name_list:
        #     symptom_flag |= self.judge_wd_symptom()
        return symptom_flag

    def _if_wd_symptom(self, li, expect: str):
        if li is None:
            return False
        if len(li) >= self.window_size * 2:  # 不然第二段因为数量少导致var必然小
            s = self.window_size
            mean_now = get_ave(li[-s:])
            var_now = float(torch.var(torch.tensor(li[-s:])))
            mean_pre = get_ave(li[-s * 2:-s])
            var_pre = float(torch.var(torch.tensor(li[-s * 2:-s])))

            if var_now > var_pre:  # 首先要求稳定性
                return True
            if expect == "inc" and mean_now <= mean_pre:
                return True
            if expect == "dec" and mean_now >= mean_pre:
                return True
        return False

    def if_wd_symptom(self, s):
        if s == "acc":
            # return self._if_wd_symptom(self.acc_list, "inc") and self._if_wd_symptom(self.val_acc_list, "inc")
            return self._if_wd_symptom(self.acc_list, "inc")
        elif s == "loss":
            # return self._if_wd_symptom(self.loss_list, "dec") and self._if_wd_symptom(self.val_loss_list, "dec")
            return self._if_wd_symptom(self.loss_list, "inc")
        elif s == "reward":
            # return self._if_wd_symptom(self.reward_list, "inc") and self._if_wd_symptom(self.val_reward_list, "inc")
            return self._if_wd_symptom(self.reward_list, "inc")

    def if_dd_et(self):
        if not self.if_enable(["model"]):
            return False
        self.cnt_et_count = self.cnt_et_count + 1 if self.epoch_has_nan_inf else 0
        return self.cnt_et_count >= self.continuous_trigger_count

    def if_dd_uw(self):
        if not self.if_enable(["model"]):
            return False
        if self.epoch_idx + 1 >= self.window_size:
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
        if self.epoch_idx + 1 <= self.window_size:
            return False
        symptom_flag = True if loss_list[-1] >= max(loss_list[-self.window_size:]) else False
        self.cnt_lnd_count = self.cnt_lnd_count + 1 if symptom_flag else 0
        return self.cnt_lnd_count >= self.continuous_trigger_count

    def if_dd_lnd(self):
        if self.if_enable(["val"]):
            tmp = self.cnt_lnd_count
            symptom_flag1 = self._if_dd_lnd(self.loss_list)
            cnt_lnd_count1 = self.cnt_lnd_count

            self.cnt_lnd_count = tmp
            symptom_flag2 = self._if_dd_lnd(self.val_loss_list)
            cnt_lnd_count2 = self.cnt_lnd_count

            self.cnt_lnd_count = max(cnt_lnd_count1, cnt_lnd_count2)
            return symptom_flag1 or symptom_flag2
        else:
            return self._if_dd_lnd(self.loss_list)

    def _if_dd_ani(self, acc_list):
        if not self.if_enable(["acc"]):
            return False
        if self.epoch_idx + 1 <= self.window_size:
            return False
        symptom_flag = True if acc_list[-1] <= min(acc_list[-self.window_size:]) else False
        self.cnt_ani_count = self.cnt_ani_count + 1 if symptom_flag else 0
        return self.cnt_ani_count >= self.continuous_trigger_count

    def if_dd_ani(self):
        if self.if_enable(["val"]):
            tmp = self.cnt_ani_count
            symptom_flag1 = self._if_dd_ani(self.acc_list)
            cnt_ani_count1 = self.cnt_ani_count

            self.cnt_ani_count = tmp
            symptom_flag2 = self._if_dd_ani(self.val_acc_list)
            cnt_ani_count2 = self.cnt_ani_count

            self.cnt_ani_count = max(cnt_ani_count1, cnt_ani_count2)
            return symptom_flag1 or symptom_flag2
        else:
            return self._if_dd_ani(self.acc_list)

    def if_dd_vg(self):
        if not self.if_enable(["model"]):
            return False
        # symptom_flag = False
        # count = 0
        # for item in self.weight_grad_abs_avg_1da:
        #     if item < self.dp.dd_threshold_VG:
        #         count += 1
        # if count / len(self.module_name_list) >= 1/2: #####
        #     symptom_flag = True
        symptom_flag = False
        for item in self.weight_grad_abs_avg_1da:
            if item < self.dp.dd_threshold_VG:
                symptom_flag = True
                break
        # cnt_vg2_count
        self.cnt_vg2_count = self.cnt_vg2_count + 1 if symptom_flag else 0
        return self.cnt_vg2_count >= self.continuous_trigger_count

    def _get_partial_ave(self, i_lst):
        v_lst = self.weight_grad_abs_avg_1da
        tmp_lst = []
        for i in range(len(v_lst)):
            if i in i_lst:
                tmp_lst.append(v_lst[i])
        return sum(tmp_lst) / len(tmp_lst)

    def if_vg(self):
        if not self.if_enable(["acc", "model"]):
            return False

        symptom_flag = False
        if 0 in self.weight_grad_abs_avg_1da:
            symptom_flag = True
        else:
            for module_name_flow_list in self.module_name_flow_matrix:
                # module_idx_flow_list = [self.module_name_list.index(name) for name in module_name_flow_list]
                ###############
                module_idx_list_flow_list = []
                for name in module_name_flow_list:
                    idx_lst = []
                    for idx in range(len(self.module_name_list)):
                        full_name = self.module_name_list[idx]
                        if name in full_name and full_name.index(name) == 0:  # prefix
                            idx_lst.append(idx)
                    module_idx_list_flow_list.append(idx_lst)
                logger.debug(" ".join(["module_idx_list_flow_list:", str(module_idx_list_flow_list)]))
                if self.last_acc <= self.dp.theta and \
                        self._get_partial_ave(module_idx_list_flow_list[0]) <= self.dp.beta2:
                    for i in range(len(module_idx_list_flow_list) - 1):
                        idx_1st1 = module_idx_list_flow_list[i]
                        idx_lst2 = module_idx_list_flow_list[i + 1]
                        v1 = self._get_partial_ave(idx_1st1)
                        v2 = self._get_partial_ave(idx_lst2)
                        if v1 / v2 <= self.dp.beta1:
                            symptom_flag = True
                            break
                    if symptom_flag:
                        break
        self.cnt_vg_count = self.cnt_vg_count + 1 if symptom_flag else 0
        return True if self.cnt_vg_count >= self.continuous_trigger_count else False

    def if_eg(self):
        if not self.if_enable(["acc", "model"]):
            return False
        symptom_flag = False
        if self.epoch_has_nan_inf:
            symptom_flag = True  ####
        else:
            for module_name_flow_list in self.module_name_flow_matrix:
                ###############
                module_idx_list_flow_list = []
                for name in module_name_flow_list:
                    idx_lst = []
                    for idx in range(len(self.module_name_list)):
                        full_name = self.module_name_list[idx]
                        if name in full_name and full_name.index(name) == 0:  # prefix
                            idx_lst.append(idx)
                    module_idx_list_flow_list.append(idx_lst)
                logger.debug(" ".join(["module_idx_list_flow_list:", str(module_idx_list_flow_list)]))
                if self.last_acc <= self.dp.theta:
                    for i in range(len(module_idx_list_flow_list) - 1):
                        idx_1st1 = module_idx_list_flow_list[i]
                        idx_lst2 = module_idx_list_flow_list[i + 1]
                        v1 = self._get_partial_ave(idx_1st1)
                        v2 = self._get_partial_ave(idx_lst2)
                        if v1 / v2 >= self.dp.beta3:
                            symptom_flag = True
                            break
                    if symptom_flag:
                        break
        self.cnt_eg_count = self.cnt_eg_count + 1 if symptom_flag else 0
        return True if self.cnt_eg_count >= self.continuous_trigger_count else False

    def if_dr(self):
        if not self.if_enable(["acc", "model"]):
            return False

        self.cnt_dr_count += 1 if self.last_acc <= self.dp.theta and self.weight_grad_rate0 >= self.dp.gamma else 0
        return True if self.cnt_dr_count >= self.continuous_trigger_count else False

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
        return count / len(acc_list) if len(acc_list) > 0 else 0

    def if_ol(self):
        # 虽然ol 但是只用了acc ol整体出现频率很低
        if not self.if_enable(["acc"]):
            return False
        ol_rate = self.get_ol_metric(self.acc_list)
        self.cnt_ol_count = self.cnt_ol_count + 1 if ol_rate >= self.dp.eta else 0
        return True if self.cnt_ol_count >= self.continuous_trigger_count else False

    def if_sc(self):
        return self._if_sc(self.acc_list)  # sc 只考虑 train acc

    def _if_sc(self, acc_list):
        # 任意的 i 都要满足 才触发！
        if not self.if_enable(["acc"]):
            return False
        if len(acc_list) <= 1:  ###
            return False
        symptom_flag = True  ###
        for i in range(1, len(acc_list)):  # !!!!! 0- -1
            if acc_list[i] - acc_list[i - 1] > self.dp.delta:
                symptom_flag = False
                break
        self.cnt_sc_count = self.cnt_sc_count + 1 if symptom_flag else 0
        return True if self.cnt_sc_count >= self.continuous_trigger_count else False


def get_methods(self):
    lst1 = list(filter(lambda m: not m.startswith("__") and callable(getattr(self, m)), dir(self)))
    print("关键方法:", lst1)
    lst2 = list(filter(lambda m: not m.startswith("__"), dir(self)))
    print("关键变量", lst2)


if __name__ == '__main__':
    get_methods(ATDDInspector)
