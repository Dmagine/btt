import logging

import numpy as np
from nni.assessor import Assessor, AssessResult

from atdd_messenger import ATDDMessenger
from atdd_utils import set_seed, diagnose_params

logger = logging.getLogger(__name__)


class ATDDAssessor(Assessor):
    def __init__(self, shared, basic, compare, diagnose, seed=None):
        super().__init__()
        self.etr = MyAssessor(shared, basic, compare, diagnose, seed)

    def assess_trial(self, trial_id, result_dict_list):
        has_symptom = self.etr.assess_trial(trial_id, result_dict_list)
        ATDDMessenger().write_assessor_info(self.etr.info_dict)
        return AssessResult.Bad if has_symptom else AssessResult.Good


class MyAssessor:
    def __init__(self, shared, basic, compare, diagnose, seed=None):
        set_seed(seed, "assessor", logger)
        self.shared = shared
        self.basic = basic
        self.compare = compare
        self.diagnose = diagnose

        self.max_epoch = self.shared["max_epoch"]
        self.enable_dict = self.shared["enable_dict"]
        self.quick_calc = self.shared["quick_calc"]
        self.start_step_float = self.basic["start_step_float"]
        self.end_step_float = self.basic["end_step_float"]
        self.symptom_name_list = self.basic["symptom_name_list"]  # ["VG", "EG", "DR", "SC", "HO", "NG"]
        self.cmp_mode = self.compare["cmp_mode"]
        self.cmp_percent = self.compare["cmp_percent"]
        self.cmp_metric_name = self.compare["cmp_metric_name"]
        self.min_cmp_num = self.compare["min_cmp_num"]
        self.dp = diagnose_params(**diagnose)

        # self.epoch_metric_3d_list = None
        # self.init_()
        self.finished_loss_list = []  # finished trial loss

        self.trial_id = None
        self.result_dict_list = None
        self.result_dict = None
        self.epoch_idx = None

        self.top_id_set = set()
        self.info_dict = None
        # "VG", "EG", "DR", "SC", "HO", "NG",(all list), "symptom_flag"(bool) or None

        ###
        self.module_metric_2da = None
        self.metric_prefix_list = None
        self.metric_suffix_list = None
        self.weight_grad_rate0_1da = None
        self.weight_grad_abs_avg_1da = None
        self.module_nele_list = None
        self.has_nan_inf_list = None

        self.has_nan_inf = None

    def assess_trial(self, trial_id, result_dict_list):
        self.trial_id = trial_id
        self.result_dict_list = result_dict_list
        self.result_dict = self.result_dict_list[-1]
        self.has_nan_inf = False

        self.info_dict = self.get_default_info_dict()
        self.receive_monitor_result()

        self.diagnose_symptom()
        return self.assess_trial_end()

    def get_default_info_dict(self):
        d = {}
        for symptom_name in self.symptom_name_list:
            d[symptom_name] = None
        d["has_symptom"] = False
        return d

    def top_performance(self):
        # top_flag = False
        # self.info_dict["top"] = []
        #
        # if self.enable_dict["acc"] is True:
        #     metric_idx = self.metric_name_list.index("train_acc")
        #     acc_history_list = self.epoch_metric_3d_list[self.epoch_idx][metric_idx]
        #     if len(acc_history_list) > self.comparable_trial_minimum:
        #         acc_t = np.percentile(acc_history_list, self.top_percent)
        #         acc = self.get_metric("train_acc")
        #         if acc is not None and acc > acc_t:
        #             s = " ".join(["top acc:", str(self.epoch_idx), self.trial_id, str(acc), ">", str(acc_t)])
        #             print(s)
        #             logger.info(s)
        #             self.info_dict["top"].append("train_acc")
        #             top_flag = True
        # if self.enable_dict["loss"] is True:
        #     metric_idx = self.metric_name_list.index("train_loss")
        #     loss_history_list = self.epoch_metric_3d_list[self.epoch_idx][metric_idx]
        #     if len(loss_history_list) > self.comparable_trial_minimum:
        #         loss_t = np.percentile(loss_history_list, 100 - self.top_percent)
        #         loss = self.get_metric("train_loss")
        #         if loss is not None and loss < loss_t:
        #             s = " ".join(["top loss:", str(self.epoch_idx), self.trial_id, str(loss), "<", str(loss_t)])
        #             logger.info(s)
        #             self.info_dict["top"].append("train_loss")
        #             top_flag = True
        # if top_flag:
        #     self.top_id_set.add(self.trial_id)
        # self.info_dict["top"] = self.info_dict["top"] if len(self.info_dict["top"]) != 0 else None
        # return top_flag
        pass

    def get_metric(self, metric_name):
        if metric_name == "train_acc":
            return self.result_dict["train_acc"] if type(self.result_dict["train_acc"]) is float else None
        elif metric_name == "train_loss":
            return self.result_dict["train_loss"] if type(self.result_dict["train_loss"]) is float else None
        else:
            return 0
        pass

    def assess_trial_end(self):
        symptom_flag = False
        for symptom_name in self.symptom_name_list:
            if self.info_dict[symptom_name] is not None:  ###
                symptom_flag = True
        if self.trial_id in self.top_id_set:
            symptom_flag = False
        self.info_dict.update({"symptom_flag": symptom_flag})

        train_loss = self.get_metric("train_loss")
        if train_loss is not None and (symptom_flag or len(self.result_dict_list) == self.max_epoch):
            self.finished_loss_list.append(train_loss)
        return symptom_flag

    def receive_monitor_result(self):
        def get_metric_array(p, s):
            idx = self.metric_prefix_list.index(p) * len(self.metric_suffix_list) + self.metric_suffix_list.index(s)
            return self.module_metric_2da[:, idx].flatten()

        d = self.result_dict
        self.epoch_idx = self.result_dict["epoch_idx"]
        self.module_nele_list = d["module_nele_list"]
        self.has_nan_inf_list = d["has_nan_inf_list"]
        if self.quick_calc:
            self.weight_grad_abs_avg_1da = d["weight_grad_abs_avg_1da"]
            self.weight_grad_rate0_1da = d["weight_grad_rate0_1da"]
        else:
            self.module_metric_2da = d["module_metric_2da"]
            self.metric_prefix_list = d["metric_prefix_list"]
            self.metric_suffix_list = d["metric_suffix_list"]
            self.weight_grad_rate0_1da = get_metric_array("weight_grad_rate", "rate0")
            self.weight_grad_abs_avg_1da = get_metric_array("weight_grad_abs", "avg")

        if type(self.weight_grad_abs_avg_1da) is dict:
            self.weight_grad_abs_avg_1da = np.array(self.weight_grad_abs_avg_1da["__ndarray__"])
            self.weight_grad_rate0_1da = np.array(self.weight_grad_rate0_1da["__ndarray__"])

    def if_loss_vulnerable(self, loss):
        global_loss_list = self.finished_loss_list
        if len(global_loss_list) < self.min_cmp_num:
            return False
        # print("global_loss_list:", global_loss_list)
        loss_t = np.percentile(global_loss_list, self.cmp_percent)  # idx越大val越大
        if loss > loss_t:
            return True
        return False

    def diagnose_symptom(self):
        self.diagnose_eg()
        if self.has_nan_inf:
            return
        self.diagnose_vg()
        self.diagnose_dr()
        self.diagnose_sc()
        self.diagnose_ho()
        self.diagnose_ng()

    def diagnose_eg(self):
        # EG: (step:half1)
        # eg_rule0: any(has_nan or has_inf)
        # eg_rule1: max(grad_abs_ave) > p_eg1 ||| (p_eg1:10)
        # eg_rule2: max(adjacent_quotient) > p_eg2 ||| (p_eg2:1000)
        # eg_rule3: loss >= cmp_median_loss * p_eg3 ||| (p_eg3:10)
        self.info_dict["EG"] = []
        self.eg_rule0()
        if self.has_nan_inf:
            pass
        else:
            self.eg_rule1()
            self.eg_rule2()
            self.eg_rule3()
        self.info_dict["EG"] = self.info_dict["EG"] if len(self.info_dict["EG"]) != 0 else None

    def eg_rule0(self):
        symptom_flag = False
        if True in self.has_nan_inf_list:
            symptom_flag = True
        if self.enable_dict["model"] and type(self.get_metric("train_loss")) is None:
            symptom_flag = True
        if self.enable_dict["model"] and type(self.weight_grad_abs_avg_1da[0]) is not np.float64:
            symptom_flag = True
        self.has_nan_inf = symptom_flag
        return symptom_flag

    def eg_rule1(self):
        if self.enable_dict["model"] is False:
            return
        symptom_flag = False
        if np.max(self.weight_grad_abs_avg_1da) > self.dp.p_eg1:
            symptom_flag = True
        if symptom_flag:
            self.info_dict["EG"].append("eg_rule1")

    def eg_rule2(self):
        if self.enable_dict["model"] is False:
            return
        symptom_flag = False
        adjacent_quotient_list = self.weight_grad_abs_avg_1da[:-1] / self.weight_grad_abs_avg_1da[1:]  ####
        if True in np.isnan(adjacent_quotient_list) or True in np.isinf(adjacent_quotient_list):
            print("adjacent_quotient_list:", adjacent_quotient_list)
            symptom_flag = True
        elif np.max(adjacent_quotient_list) > self.dp.p_eg2:
            symptom_flag = True
        if symptom_flag:
            self.info_dict["EG"].append("eg_rule2")

    def eg_rule3(self):
        if self.enable_dict["model"] is False:
            return
        symptom_flag = False
        last_loss = self.result_dict["loss"]
        global_loss_list = self.finished_loss_list
        if len(global_loss_list) < self.min_cmp_num:
            return
        median_global_loss = np.median(global_loss_list)
        if last_loss > median_global_loss * self.dp.p_eg3:
            symptom_flag = True
        if symptom_flag:
            self.info_dict["EG"].append("eg_rule3")

    def diagnose_vg(self):
        # VG: (step:half1)
        # protect_top_loss: True
        # vg_rule1: median(grad_abs_ave) < p_vg1 ||| (p_vg1:1.e-7) 。。。魔法数，考虑取消half？？？
        # vg_rule2: median(adjacent_quotient) < p_vg2 ||| (p_vg3:0.001) 逻辑修复+已经改大p_vg3
        # vg_rule3: count(delta_loss=0) / max_epoch >= p_vg3 ||| (p_vg3:0.1)
        self.info_dict["VG"] = []
        if self.if_loss_vulnerable(self.get_metric("train_loss")):
            self.vg_rule1()
            self.vg_rule2()
            self.vg_rule3()
        self.info_dict["VG"] = self.info_dict["VG"] if len(self.info_dict["VG"]) != 0 else None

    def vg_rule1(self):
        if self.enable_dict["model"] is False:
            return
        symptom_flag = False
        if np.median(self.weight_grad_abs_avg_1da) < self.dp.p_vg1:
            symptom_flag = True
        if symptom_flag:
            self.info_dict["VG"].append("vg_rule1")

    def vg_rule2(self):
        if self.enable_dict["model"] is False:
            return
        symptom_flag = False
        adjacent_quotient_list = self.weight_grad_abs_avg_1da[:-1] / self.weight_grad_abs_avg_1da[1:]  ####
        if True in np.isnan(adjacent_quotient_list) or True in np.isinf(adjacent_quotient_list):
            symptom_flag = True
        elif np.median(adjacent_quotient_list) < self.dp.p_vg2:
            symptom_flag = True
        if symptom_flag:
            self.info_dict["VG"].append("vg_rule2")

    def vg_rule3(self):
        if self.enable_dict["loss"] is False:
            return
        symptom_flag = False
        loss_list = np.array(self.result_dict["train_loss_list"])
        if len(loss_list) < 2:
            return
        delta_loss_list = loss_list[1:] - loss_list[:-1]
        if np.sum(delta_loss_list == 0) / self.max_epoch >= self.dp.p_vg3:
            symptom_flag = True
        if symptom_flag:
            self.info_dict["VG"].append("vg_rule3")

    def diagnose_dr(self):
        # DR: (step:all)
        # protect_top_loss: True
        # dr_rule1: median(rate0) < p_dr1 ||| (p_dr1:0.1) 已经改any为median且调小了p_dr1
        # dr_rule2: weighted_mean(rate0) > p_dr2 ||| (p_dr2:0.5) 。。。同质，考虑取消
        self.info_dict["DR"] = []
        if self.if_loss_vulnerable(self.get_metric("train_loss")):
            self.dr_rule1()
            # self.dr_rule2()
        self.info_dict["DR"] = self.info_dict["DR"] if len(self.info_dict["DR"]) != 0 else None

    def dr_rule1(self):
        if self.enable_dict["model"] is False:
            return
        if np.median(self.weight_grad_rate0_1da) > self.dp.p_dr1:
            self.info_dict["DR"].append("dr_rule1")

    def dr_rule2(self):
        # if self.enable_dict["model"] is False:
        #     return
        # if np.average(self.weight_grad_rate0_1da, weights=self.module_nele_list) > self.dp.p_dr2:
        #     self.info_dict["DR"].append("dr_rule2")
        pass

    def diagnose_sc(self):
        # SC: (step:all)
        # protect_top_loss: True
        # sc_rule1: (acc[0]-acc[-1]) / acc[0] > p_sc1 ||| (p_sc1:0) 。。。同质acc混，考虑取消
        # sc_rule2: (loss[-1]-loss[0]) / loss[0] > p_sc2 ||| (p_sc2:0) 已经改逻辑，只避免比初始越差
        self.info_dict["SC"] = []
        if self.if_loss_vulnerable(self.get_metric("train_loss")):
            # self.sc_rule1()
            self.sc_rule2()
        self.info_dict["SC"] = self.info_dict["SC"] if len(self.info_dict["SC"]) != 0 else None

    def sc_rule1(self):
        # if self.enable_dict["acc"] is False:
        #     return
        # acc_list = np.array(self.result_dict["train_acc_list"])
        # start_step = 2
        # end_step = int(self.max_epoch / 2)
        # if len(acc_list) > end_step or len(acc_list) < start_step:
        #     return
        # if (acc_list[-1] - acc_list[0]) / len(acc_list) < (1 - acc_list[0]) / self.max_epoch * self.dp.p_sc1:
        #     self.info_dict["SC"].append("sc_rule1")
        pass

    def sc_rule2(self):
        if self.enable_dict["loss"] is False:
            return
        loss_list = np.array(self.result_dict["train_loss_list"])
        start_step = 2
        end_step = int(self.max_epoch / 2)
        if len(loss_list) > end_step or len(loss_list) < start_step:
            return
        if (loss_list[-1] - loss_list[0]) / loss_list[0] > self.dp.p_sc2:
            self.info_dict["SC"].append("sc_rule2")

    def diagnose_ho(self):
        # HO(heavy oscillation): (step:half2) (wd:0.25) 。。。。斜率配合MAE
        # ho_rule1: std(acc[-wd:]) / mean(acc[-wd:]) > p_ho1 ||| (p_ho1:0) 。。。同质acc混，考虑取消
        # ho_rule2: MAE(loss[-wd:] - line(loss[-wd:])) > mean(loss[-wd:]) * p_ho2  ||| (p_ho2:0.1) 已经改逻辑，有待验证
        self.info_dict["HO"] = []
        # self.ho_rule1()
        self.ho_rule2()
        self.info_dict["HO"] = self.info_dict["HO"] if len(self.info_dict["HO"]) != 0 else None

    def ho_rule1(self):
        # if self.enable_dict["acc"] is False:
        #     return
        # acc_list = np.array(self.result_dict["train_acc_list"])
        # window_size = int(round(self.dp.wd * self.max_epoch))
        # start_step = int(self.max_epoch / 2)
        # end_step = self.max_epoch
        # if len(acc_list) > end_step or len(acc_list) < start_step:
        #     return
        # sub_list = acc_list[-window_size:]
        # if np.std(sub_list) / np.mean(sub_list) > self.dp.p_ho1:
        #     self.info_dict["HO"].append("ho_rule1")
        pass

    def ho_rule2(self):
        if self.enable_dict["loss"] is False:
            return
        loss_list = np.array(self.result_dict["train_loss_list"])
        window_size = int(round(self.dp.wd * self.max_epoch))
        start_step = int(self.max_epoch / 2)
        end_step = self.max_epoch
        if len(loss_list) > end_step or len(loss_list) < start_step:
            return
        sub_list = loss_list[-window_size:]
        line = np.polyfit(np.arange(len(sub_list)), sub_list, 1)
        mae = np.mean(np.abs(sub_list - np.polyval(line, np.arange(len(sub_list)))))
        # print(mae, np.mean(sub_list), np.polyval(line, np.arange(len(sub_list))), sub_list)
        if mae > np.mean(sub_list) * self.dp.p_ho2:
            self.info_dict["HO"].append("ho_rule2")

    def diagnose_ng(self):
        # NG(no gain): (step:half2) (wd:0.25)
        # protect_top_loss: True
        # ng_rule1: max(acc[-wd:]) != max(acc) 。。。同质acc混，考虑取消
        # ng_rule2: min(loss[-wd:]) != min(loss)
        self.info_dict["NG"] = []
        if self.if_loss_vulnerable(self.get_metric("train_loss")):
            # self.ng_rule1()
            self.ng_rule2()
        self.info_dict["NG"] = self.info_dict["NG"] if len(self.info_dict["NG"]) != 0 else None

    def ng_rule1(self):
        # if self.enable_dict["acc"] is False:
        #     return
        # acc_list = np.array(self.result_dict["train_acc_list"])
        # start_step = int(self.max_epoch / 2)
        # end_step = self.max_epoch
        # window_size = int(round(self.dp.wd * self.max_epoch))
        # if len(acc_list) > end_step or len(acc_list) < start_step:
        #     return
        # if np.max(acc_list[-window_size:]) != np.max(acc_list):
        #     self.info_dict["NG"].append("ng_rule1")
        pass

    def ng_rule2(self):
        if self.enable_dict["loss"] is False:
            return
        loss_list = np.array(self.result_dict["train_loss_list"])
        start_step = int(self.max_epoch / 2)
        end_step = self.max_epoch
        window_size = int(round(self.dp.wd * self.max_epoch))
        if len(loss_list) > end_step or len(loss_list) < start_step:
            return
        if np.min(loss_list[-window_size:]) != np.min(loss_list):
            self.info_dict["NG"].append("ng_rule2")
