import logging

import numpy as np
from nni.assessor import Assessor, AssessResult

from atdd_messenger import ATDDMessenger
from atdd_utils import set_seed

logger = logging.getLogger(__name__)


class ATDDAssessor(Assessor):
    def __init__(self, shared, basic, compare, diagnose, seed=None):
        super().__init__()
        set_seed(seed, "assessor", logger)
        self.shared = shared
        self.basic = basic
        self.compare = compare
        self.diagnose = diagnose
        self.complete_config_by_default()

        self.model_num = self.shared["model_num"]
        self.enable_dict = self.shared["enable_dict"]
        self.max_epoch = self.basic["max_epoch"]
        self.maximize_metric_name_list = self.basic["maximize_metric_name_list"]
        self.minimize_metric_name_list = self.basic["minimize_metric_name_list"]
        self.k = self.compare["k"]
        self.max_percentile = self.compare["max_percentile"]  # eg. 90
        self.start_step_float = self.compare["start_step_float"]
        self.end_step_float = self.compare["end_step_float"]
        self.comparable_trial_minimum = self.compare["comparable_trial_minimum"]
        self.metric_demarcation_num = self.compare["metric_demarcation_num"]  # boundary 避免太紧 suanle
        self.beta1 = self.diagnose["beta1"]
        self.beta3 = self.diagnose["beta3"]
        self.zeta = self.diagnose["zeta"]
        self.window_size_float = self.diagnose["window_size_float"]

        self.cur_step = 0

        self.metric_num = self.get_cmp_metric_num()
        self.cmp_step_list = None  # [1, np.ceil(s / 2), s, 2 * s]
        self.cmp_percentile_list = None  # [3.125, 12.5, 25, 50]  # (0,100) for np.percentile() 25下四分卫 75上四分卫
        self.init_cmp_list()

        self.step_metric_score_list_dict_dict = None  # history
        self.init_history()
        # self.result_dict_list_dict = {}

        self.messenger = None
        # top
        self.result_dict = None
        self.step_acc_loss_list_dict = None
        return

    def complete_config_by_default(self):
        pass

    def init_history(self):
        self.step_metric_score_list_dict_dict = {}
        for step in self.cmp_step_list:
            self.step_metric_score_list_dict_dict[step] = {}
            for metric_name in self.maximize_metric_name_list + self.minimize_metric_name_list:
                self.step_metric_score_list_dict_dict[step][metric_name] = []

    def record_useful_history(self, step, metric_score_dict):
        for metric_name, score in metric_score_dict.items():
            self.step_metric_score_list_dict_dict[step][metric_name].append(score)

    def init_cmp_list(self):
        max_epoch = self.max_epoch
        step_list = []
        percentile_list = []

        start_s = int(round(self.start_step_float * max_epoch))
        end_s = int(round(self.end_step_float * max_epoch))
        step = end_s
        while True:
            if step < start_s or step in step_list:
                break
            step_list.insert(0, step)
            percentile_list.insert(0, min(self.max_percentile, 100 * step / max_epoch * self.k))
            step = int(np.ceil((step - start_s) / 2))

        pre, now = 0, 0
        pass_list = []
        for val in percentile_list:
            now = pre + (100 - pre) * val / 100
            pass_list.append(now)
            pre = now

        logger.info("step_list: " + str(step_list))
        logger.info("percentile_list: " + str(percentile_list))
        logger.info("pass_list: " + str(pass_list))
        # [1, 2, 3, 5, 10]
        # [5.0, 10.0, 15.0, 25.0, 50.0]

        self.cmp_step_list = step_list
        self.cmp_percentile_list = percentile_list

    def get_veg_metric(self, param_grad_abs_ave_list, module_name_flow_2dlist, module_name_list):
        if 0 in param_grad_abs_ave_list:
            return np.log10(self.beta3) - np.log10(self.beta1)
        else:
            mid = (np.log10(self.beta3) - np.log10(self.beta1))
            lst = []
            for module_name_flow_list in module_name_flow_2dlist:
                module_idx_flow_list = [module_name_list.index(name) for name in module_name_flow_list]
                for i in range(len(module_idx_flow_list) - 1):
                    idx_1 = module_idx_flow_list[i]
                    idx_2 = module_idx_flow_list[i + 1]
                    v1, v2 = param_grad_abs_ave_list[idx_1], param_grad_abs_ave_list[idx_2]
                    lst.append(abs(np.log10(v1 / v2) - mid))
            return sum(lst) / len(lst)

    def get_ol_metric(self, acc_list):
        # dist = 0
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
            # dist += maximum_list[i] - minimum_list[i]
            if maximum_list[i] - minimum_list[i] >= self.zeta:
                count += 1
        # return [count / len(acc_list), dist / len(acc_list)]
        return count / len(acc_list)

    def get_metric_value(self, result_dict, metric_name):
        d = result_dict
        if metric_name in ["acc", "loss", "reward", "val_acc", "val_loss", "val_reward"]:
            return d[metric_name]
        if metric_name == "veg_metric":
            return self.get_veg_metric(d["param_grad_abs_ave_list"], d["module_name_flow_2dlist"],
                                       d["module_name_list"])
        if metric_name == "dr_metric":
            return d["param_grad_zero_rate"]
        if metric_name == "ol_metric":
            return self.get_ol_metric(d["acc_list"])
        if metric_name == "sc_metric":
            return d["acc_list"][-1] - d["acc_list"][0]

    def get_metric_window_ave(self, metric_name, result_dict_list, step_end):
        step_start = max(0, step_end - int(round(self.window_size_float * self.max_epoch)))
        ll = []
        for i in range(step_start, step_end):  # (2,5) -> 2 3 4
            ##############
            val = self.get_metric_value(result_dict_list[i], metric_name)
            # val = result_dict_list[i][metric_name]
            ll.append(val)
            if np.isnan(val) or np.isinf(val):
                return None
        return sum(ll) / len(ll)

    def get_cmp_metric_num(self):
        count = 0
        for lst in [self.maximize_metric_name_list, self.minimize_metric_name_list]:
            count += len(lst) if lst is not None else 0
        return count

    def get_default_dict(self):
        d = {"step_counter": self.cur_step}
        for metric_name in self.minimize_metric_name_list + self.maximize_metric_name_list:
            d.update({("cmp_" + metric_name): None})
        return d

    def send_msg(self, metric_msg_dict=None):  # weak or bad
        d = self.get_default_dict()
        if metric_msg_dict is not None:
            d.update(metric_msg_dict)
        self.messenger.write_assessor_info(d)
        return AssessResult.Good

    def top_performance(self):
        if self.step_acc_loss_list_dict is None:
            self.step_acc_loss_list_dict = {}  # step
            for step in self.cmp_step_list:
                self.step_acc_loss_list_dict.update({step: {"acc_list": [], "loss_list": []}})
        acc_list = self.result_dict["acc_list"]
        loss_list = self.result_dict["loss_list"]

        if acc_list is not None and len(acc_list) > 0:
            self.step_acc_loss_list_dict[self.cur_step]["acc_list"].append(acc_list[-1])
            d = self.step_acc_loss_list_dict
            if acc_list[-1] > np.percentile(d[self.cur_step]["acc_list"], 95):
                logger.info(" ".join(["top acc performance:", self.result_dict["trial_id"], str(acc_list[-1])]))
                return True
        if loss_list is not None and len(loss_list) > 0:
            self.step_acc_loss_list_dict[self.cur_step]["loss_list"].append(loss_list[-1])
            d = self.step_acc_loss_list_dict
            if loss_list[-1] < np.percentile(d[self.cur_step]["loss_list"], 5):
                logger.info(" ".join(["top loss performance:", self.result_dict["trial_id"], str(loss_list[-1])]))
                return True
        return False

    def assess_trial(self, trial_id, result_dict_list):
        """
        Determines whether a trial should be killed. Must override.
        trial_history: a list of intermediate result objects.
        Returns AssessResult.Good or AssessResult.Bad.
        """
        # log
        self.messenger = ATDDMessenger(trial_id)
        self.result_dict = dict(result_dict_list[-1])
        logger.info("send intermediate_result_dict: %s: %s" % (trial_id, self.result_dict["step_counter"]))
        logger.debug("intermediate_result_dict: %s: %s" % (trial_id, str(self.result_dict)))

        cur_step = len(result_dict_list)
        self.cur_step = cur_step
        if cur_step not in self.cmp_step_list:
            return self.send_msg()

        if self.top_performance():
            return self.send_msg()

        # calculate this trial
        metric_score_dict = {}  # metric name -> metric value window ave
        bad_metric_msg_dict = {}
        for metric_name in self.minimize_metric_name_list + self.maximize_metric_name_list:
            metric_val = self.get_metric_window_ave(metric_name, result_dict_list, cur_step)
            if metric_val is None:  # nan
                bad_metric_msg_dict.update({"cmp_" + metric_name: "inf_or_nan"})
                continue
            metric_score_dict.update({metric_name: metric_val})
        if len(bad_metric_msg_dict) != 0:
            self.record_useful_history(cur_step, metric_score_dict)  # no nan
            logger.info(" ".join(["early stop inf_or_nan:", trial_id, str(cur_step)]))
            logger.debug(" ".join(["Early Stopped:", trial_id, str(cur_step), str(bad_metric_msg_dict), "\n"]))
            return self.send_msg(bad_metric_msg_dict)

        # # 17 [5,10] [[1,2,3,4,5] [6,7,8,9,10]] -> [[0,1,2,3,4] [5,6,7,8,9]] (索引)

        metric_score_list_dict = self.step_metric_score_list_dict_dict[cur_step]  # not include now
        self.record_useful_history(cur_step, metric_score_dict)

        # compare
        out_dict = {}
        for metric_name in self.maximize_metric_name_list + self.minimize_metric_name_list:
            cmp_trial_num = len(metric_score_list_dict[metric_name])
            if cmp_trial_num < self.comparable_trial_minimum:
                logger.info(" ".join(["limited cmp_trial_num:", trial_id, metric_name]))
                return self.send_msg()
            tmp = self.cmp_step_list.index(cur_step)
            if metric_name in self.maximize_metric_name_list:
                percent = self.cmp_percentile_list[tmp]
                score_1 = metric_score_dict[metric_name]
                score_2 = np.percentile(metric_score_list_dict[metric_name], percent)
                out_dict[metric_name] = {}
                out_dict[metric_name]["flag"] = True if score_1 >= score_2 else False
                out_dict[metric_name]["score_1"] = score_1
                out_dict[metric_name]["score_2"] = score_2
                # np.percentile([4,3,5,2,1],25) -> 2.0
            elif metric_name in self.minimize_metric_name_list:
                percent = 100 - self.cmp_percentile_list[tmp]  # np.percentile([1,2,5,3,4],75) -> 4
                score_1 = metric_score_dict[metric_name]
                score_2 = np.percentile(metric_score_list_dict[metric_name], percent)
                out_dict[metric_name] = {}
                out_dict[metric_name]["flag"] = True if score_1 <= score_2 else False
                out_dict[metric_name]["score_1"] = score_1
                out_dict[metric_name]["score_2"] = score_2

        # 紧 。。。。。。
        weak_metric_msg_dict = {}  # weak
        for metric_name, d in out_dict.items():
            if d["flag"] is False:
                weak_metric_msg_dict.update({"cmp_" + metric_name: "weak"})
        if (self.metric_num > self.metric_demarcation_num and len(weak_metric_msg_dict) >= self.metric_demarcation_num) \
                or (self.metric_num <= self.metric_demarcation_num and len(weak_metric_msg_dict) != 0):
            logger.info(" ".join(["early stop weak:", trial_id, str(cur_step)]))
            logger.debug(" ".join(["Early Stopped:", trial_id, str(cur_step),
                                   str(weak_metric_msg_dict), str(out_dict), "\n"]))
            d = weak_metric_msg_dict.copy()
            d.update({"out_dict_str": str(out_dict)})
            return self.send_msg(d)

        return self.send_msg()
