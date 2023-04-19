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
        self.symptom_name_list = self.basic["rule_name_list"]  # ["VG", "EG", "DR", "SC", "HO", "NG"]
        self.metric_name_list = self.compare["metric_name_list"]  # ["train_acc","train_loss"]
        self.top_percent = self.compare["top_percent"]
        self.comparable_trial_minimum = self.compare["comparable_trial_minimum"]
        self.dp = diagnose_params(**diagnose)

        self.epoch_metric_3d_list = None
        self.init_()

        self.trial_id = None
        self.result_dict_list = None
        self.result_dict = None
        self.epoch_idx = None

        self.top_id_set = set()
        self.info_dict = None
        # "VG", "EG", "DR", "SC", "HO", "NG",(all list), "top"(list),"has_symptom"(bool) or None

        ###
        self.module_metric_2da = None
        self.metric_prefix_list = None
        self.metric_suffix_list = None
        self.weight_grad_rate0_1da = None
        self.weight_grad_abs_avg_1da = None
        self.module_nele_list = None

    def init_(self):
        self.metric_name_list.remove("train_acc") if self.enable_dict["acc"] is False else None
        self.metric_name_list.remove("train_loss") if self.enable_dict["loss"] is False else None
        self.epoch_metric_3d_list = [[[]] * len(self.metric_name_list)] * self.max_epoch

    def assess_trial(self, trial_id, result_dict_list):
        self.trial_id = trial_id
        self.result_dict_list = result_dict_list
        self.result_dict = self.result_dict_list[-1]

        self.info_dict = self.get_default_info_dict()
        self.receive_monitor_result()
        self.record_cmp_metric()

        if self.top_performance():
            return self.end_assess()

        self.diagnose_symptom()
        return self.end_assess()

    def get_default_info_dict(self):
        d = {}
        for symptom_name in self.symptom_name_list:
            d[symptom_name] = None
        d["top"] = None
        d["has_symptom"] = False
        return d

    def top_performance(self):
        top_flag = False
        self.info_dict["top"] = []
        if self.trial_id in self.top_id_set:
            logger.info(" ".join(["top prev:", str(self.trial_id)]))
            top_flag = True
        if self.enable_dict["acc"] is True:
            metric_idx = self.metric_name_list.index("train_acc")
            acc_history_list = self.epoch_metric_3d_list[self.epoch_idx][metric_idx]
            if len(acc_history_list) > self.comparable_trial_minimum:
                acc_threshold = np.percentile(acc_history_list, self.top_percent)
                acc = self.result_dict["train_acc"]
                if acc > acc_threshold:
                    logger.info(" ".join(["top acc:", str(self.trial_id), str(acc), ">", str(acc_threshold)]))
                    self.info_dict["top"].append("train_acc")
                    top_flag = True
        if self.enable_dict["loss"] is True:
            metric_idx = self.metric_name_list.index("train_loss")
            loss_history_list = self.epoch_metric_3d_list[self.epoch_idx][metric_idx]
            if len(loss_history_list) > self.comparable_trial_minimum:
                loss_threshold = np.percentile(loss_history_list, 100 - self.top_percent)
                loss = self.result_dict["train_loss"]
                if loss < loss_threshold:
                    logger.info(" ".join(["top loss:", str(self.trial_id), str(loss), "<", str(loss_threshold)]))
                    self.info_dict["top"].append("train_loss")
                    top_flag = True
        if top_flag:
            self.top_id_set.add(self.trial_id)
        if len(self.info_dict["top"]) == 0:
            self.info_dict["top"] = None
        return top_flag

    def calc_metric(self, metric_name):
        if metric_name == "train_acc":
            return self.result_dict["train_acc"]
        elif metric_name == "train_loss":
            return self.result_dict["train_loss"]
        else:
            return 0
        pass

    def record_cmp_metric(self):
        for metric_name in self.metric_name_list:
            metric_val = self.calc_metric(metric_name)
            metric_idx = self.metric_name_list.index(metric_name)
            self.epoch_metric_3d_list[self.epoch_idx][metric_idx].append(metric_val)

    def end_assess(self):
        has_symptom = False
        for symptom_name in self.symptom_name_list:
            if self.info_dict[symptom_name] is not None:  ###
                has_symptom = True
        self.info_dict.update({"has_symptom": has_symptom})
        return has_symptom

    def receive_monitor_result(self):
        def get_metric_array(p, s):
            idx = self.metric_prefix_list.index(p) * len(self.metric_suffix_list) + self.metric_suffix_list.index(s)
            return self.module_metric_2da[:, idx].flatten()

        d = self.result_dict
        self.epoch_idx = self.result_dict["epoch_idx"]
        self.module_nele_list = d["module_nele_list"]
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

    def diagnose_symptom(self):
        self.diagnose_eg()
        self.diagnose_vg()
        self.diagnose_dr()
        self.diagnose_sc()
        self.diagnose_ho()
        self.diagnose_ng()

    def diagnose_eg(self):
        # EG: (step:half1)
        # eg_rule1: max(grad_abs_ave) > p_eg1 ||| (p_eg1:10)
        # eg_rule2: max(adjacent_quotient) > p_eg2 ||| (p_eg2:1000)
        # eg_rule3: loss >= cmp_median_loss * p_eg3 ||| (p_eg3:100)
        self.info_dict["EG"] = []
        self.eg_rule1()
        self.eg_rule2()
        self.eg_rule3()
        self.info_dict["EG"] = self.info_dict["EG"] if len(self.info_dict["EG"]) != 0 else None

    def eg_rule1(self):
        if self.enable_dict["model"] is False:
            return
        symptom_flag = False
        if True in np.isnan(self.weight_grad_abs_avg_1da) or True in np.isinf(self.weight_grad_abs_avg_1da):
            symptom_flag = True
        elif np.max(self.weight_grad_abs_avg_1da) > self.dp.p_eg1:
            symptom_flag = True
        if symptom_flag:
            self.info_dict["EG"].append("eg_rule1")

    def eg_rule2(self):
        if self.enable_dict["model"] is False:
            return
        symptom_flag = False
        adjacent_quotient_list = np.abs(self.weight_grad_abs_avg_1da[:-1], self.weight_grad_abs_avg_1da[1:])
        if True in np.isnan(adjacent_quotient_list) or True in np.isinf(adjacent_quotient_list):
            symptom_flag = True
        elif np.median(adjacent_quotient_list) > self.dp.p_eg2:
            symptom_flag = True
        if symptom_flag:
            self.info_dict["EG"].append("eg_rule2")

    def eg_rule3(self):
        if self.enable_dict["model"] is False:
            return
        symptom_flag = False
        last_loss = self.result_dict["loss"]
        metric_idx = self.metric_name_list.index("train_loss")
        median_history_loss = np.median(self.epoch_metric_3d_list[self.epoch_idx][metric_idx])
        if np.isinf(median_history_loss) or np.isnan(median_history_loss):
            symptom_flag = True
        elif last_loss > median_history_loss * self.dp.p_eg3:
            symptom_flag = True
        if symptom_flag:
            self.info_dict["EG"].append("eg_rule3")

    def diagnose_vg(self):
        # VG: (step:half1)
        # vg_rule1: median(grad_abs_ave) < p_vg1 ||| (p_vg1:1e-7)
        # vg_rule2: median(adjacent_quotient) < p_vg2 ||| (p_vg3:0.01)
        # vg_rule3: count(delta_loss=0) / max_epoch >= p_vg3 ||| (p_vg3:0.1)
        self.info_dict["VG"] = []
        self.vg_rule1()
        self.vg_rule2()
        self.vg_rule3()
        self.info_dict["VG"] = self.info_dict["VG"] if len(self.info_dict["VG"]) != 0 else None

    def vg_rule1(self):
        if self.enable_dict["model"] is False:
            return
        symptom_flag = False
        if True in np.isnan(self.weight_grad_abs_avg_1da) or True in np.isinf(self.weight_grad_abs_avg_1da):
            return
        if np.median(self.weight_grad_abs_avg_1da) < self.dp.p_vg1:
            symptom_flag = True
        if symptom_flag:
            self.info_dict["VG"].append("vg_rule1")

    def vg_rule2(self):
        if self.enable_dict["model"] is False:
            return
        symptom_flag = False
        adjacent_quotient_list = np.abs(self.weight_grad_abs_avg_1da[:-1], self.weight_grad_abs_avg_1da[1:])
        if True in np.isnan(adjacent_quotient_list) or True in np.isinf(adjacent_quotient_list):
            return
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
        if True in np.isnan(delta_loss_list) or True in np.isinf(delta_loss_list):
            return
        elif np.sum(delta_loss_list == 0) / len(delta_loss_list) >= self.dp.p_vg3:
            symptom_flag = True
        if symptom_flag:
            self.info_dict["VG"].append("vg_rule3")

    def diagnose_dr(self):
        # DR: (step:all)
        # dr_rule1: any(rate0 > p_dr1) ||| (p_dr1:0.5)
        # dr_rule2: weighted_mean(rate0) > p_dr2 ||| (p_dr2:0.5)
        self.info_dict["DR"] = []
        self.dr_rule1()
        self.dr_rule2()
        self.info_dict["DR"] = self.info_dict["DR"] if len(self.info_dict["DR"]) != 0 else None

    def dr_rule1(self):
        if self.enable_dict["model"] is False:
            return
        if True in list(self.weight_grad_rate0_1da > self.dp.p_dr1):
            self.info_dict["DR"].append("dr_rule1")

    def dr_rule2(self):
        if self.enable_dict["model"] is False:
            return
        if np.average(self.weight_grad_rate0_1da, weights=self.module_nele_list) > self.dp.p_dr2:
            self.info_dict["DR"].append("dr_rule2")

    def diagnose_sc(self):
        # SC: (step:p_sc1, p_sc2) -> (0, 0.5) min 1 epoch
        # sc_rule1: (acc[-1]-acc[0])/len(acc) < (1-acc[0])/max_epoch
        # sc_rule2: (loss[0]-loss[-1])/len(loss) < (loss[0]-0)/max_epoch
        self.info_dict["SC"] = []
        self.sc_rule1()
        self.sc_rule2()
        self.info_dict["SC"] = self.info_dict["SC"] if len(self.info_dict["SC"]) != 0 else None

    def sc_rule1(self):
        if self.enable_dict["acc"] is False:
            return
        acc_list = np.array(self.result_dict["train_acc_list"])
        start_step = 1
        end_step = int(self.max_epoch / 2)
        if len(acc_list) > end_step or len(acc_list) < start_step:
            return
        if (acc_list[-1] - acc_list[0]) / len(acc_list) < (1 - acc_list[0]) / self.max_epoch:
            self.info_dict["SC"].append("sc_rule1")

    def sc_rule2(self):
        if self.enable_dict["loss"] is False:
            return
        loss_list = np.array(self.result_dict["train_loss_list"])
        start_step = 1
        end_step = int(self.max_epoch / 2)
        if len(loss_list) > end_step or len(loss_list) < start_step:
            return
        if (loss_list[0] - loss_list[-1]) / len(loss_list) < (loss_list[0] - 0) / self.max_epoch:
            self.info_dict["SC"].append("sc_rule2")

    def diagnose_ho(self):
        # HO: (step:half2) (wd:0.25)
        # heavy oscillation
        # ho_rule1: std(acc) / mean(acc) > p_ho1 ||| (p_ho1:?)
        # ho_rule2: std(log_loss) / mean(log_loss) > p_ho2 ||| (p_ho2:?)
        self.info_dict["HO"] = []
        self.ho_rule1()
        self.ho_rule2()
        self.info_dict["HO"] = self.info_dict["HO"] if len(self.info_dict["HO"]) != 0 else None

    def ho_rule1(self):
        if self.enable_dict["acc"] is False:
            return
        acc_list = np.array(self.result_dict["train_acc_list"])
        start_step = int(self.max_epoch / 2)
        end_step = self.max_epoch
        window_size = int(round(self.dp.wd*self.max_epoch))
        if len(acc_list) > end_step or len(acc_list) < start_step:
            return
        sub_list = acc_list[-window_size:]
        if np.std(sub_list) / np.mean(sub_list) > self.dp.p_ho1:
            self.info_dict["HO"].append("ho_rule1")

    def ho_rule2(self):
        if self.enable_dict["loss"] is False:
            return
        loss_list = np.array(self.result_dict["train_loss_list"])
        start_step = int(self.max_epoch / 2)
        end_step = self.max_epoch
        window_size = int(round(self.dp.wd*self.max_epoch))
        if len(loss_list) > end_step or len(loss_list) < start_step:
            return
        sub_list = np.log(loss_list[-window_size:])
        if np.std(sub_list) / np.mean(sub_list) > self.dp.p_ho2:
            self.info_dict["HO"].append("ho_rule2")

    def diagnose_ng(self):
        # NG: (step:half2) (wd:0.25)
        # no gain
        # ng_rule1: max(acc[-wd:]) < max(acc)
        # ng_rule2: min(loss[-wd:]) < min(loss)
        # ng_metric1: acc
        # ng_metric2: loss
        self.info_dict["NG"] = []
        self.ng_rule1()
        self.ng_rule2()
        self.info_dict["NG"] = self.info_dict["NG"] if len(self.info_dict["NG"]) != 0 else None

    def ng_rule1(self):
        if self.enable_dict["acc"] is False:
            return
        acc_list = np.array(self.result_dict["train_acc_list"])
        start_step = int(self.max_epoch / 2)
        end_step = self.max_epoch
        window_size = int(round(self.dp.wd*self.max_epoch))
        if len(acc_list) > end_step or len(acc_list) < start_step:
            return
        if np.max(acc_list[-window_size:]) < np.max(acc_list):
            self.info_dict["NG"].append("ng_rule1")

    def ng_rule2(self):
        if self.enable_dict["loss"] is False:
            return
        loss_list = np.array(self.result_dict["train_loss_list"])
        start_step = int(self.max_epoch / 2)
        end_step = self.max_epoch
        window_size = int(round(self.dp.wd*self.max_epoch))
        if len(loss_list) > end_step or len(loss_list) < start_step:
            return
        if np.min(loss_list[-window_size:]) < np.min(loss_list):
            self.info_dict["NG"].append("ng_rule2")
