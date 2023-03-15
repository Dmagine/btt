import copy
import json
import logging
import random
from pathlib import Path
from typing import Union

import hyperopt as hp
import numpy as np
from nni.algorithms.hpo.hyperopt_tuner import json2vals, json2space, json2parameter, _add_index
from nni.common.hpo_utils import validate_search_space
from nni.experiment.experiment import Experiment
from nni.runtime.env_vars import _load_env_vars, _dispatcher_env_var_names
from nni.tools.nnictl.config_utils import Config
from nni.tools.nnictl.launcher_utils import parse_time
from nni.tuner import Tuner
from nni.utils import NodeType, OptimizeMode, extract_scalar_reward, split_index

from atdd_utils import set_seed

logger = logging.getLogger('atdd_tuner')


class ATDDTuner(Tuner):
    def __init__(self, shared, basic, rectify, parallel, seed=None):
        super().__init__()
        set_seed(seed, "tuner", logger=logger)

        self.shared = shared
        self.basic = basic
        self.rectify = rectify
        self.complete_config_by_default()

        self.model_num = self.shared["model_num"]
        self.enable_dict = self.shared["enable_dict"]
        self.algorithm_name = self.basic["algorithm_name"]
        self.optimize_mode = self.basic["optimize_mode"]
        self.rule_name_list = self.basic["rule_name_list"]
        self.k = self.rectify["k"]
        self.base_p = self.rectify["base_probability"]
        self.max_p = self.rectify["max_probability"]
        self.start_duration_float = self.rectify["start_duration_float"]
        self.end_duration_float = self.rectify["end_duration_float"]
        self.same_retry_maximum = self.rectify["same_retry_maximum"]
        self.retained_parameter_name_list = self.rectify["retained_parameter_list"]
        # retained_parameter_list 哪些重要参数在修复时应尽量保留

        self.exp_id = None  # for exeDuration
        self.exp = None
        self.get_exp_and_id()
        self.max_duration = None
        self.duration_list = None
        self.probability_list = None
        self.init_max_duration()
        self.init_duration_probability_list()

        # self.id_symptom_dict = {}
        # self.id_metric_dict = {}
        self.id_parameters_dict_dict = {} # key: parameter_id !!!

        # self.optimal_parameter_dict = None
        # self.optimal_parameter_metric = None
        # self.optimal_parameter_id = None
        # self.optimal_parameter_symptom_dict = None
        self.optimal_dict = None
        self.at_symptom = None
        self.dd_symptom = None
        self.wd_symptom = None
        self.data_cond = None
        self.weight_cond = None
        self.lr_cond = None
        self.old_params = None
        self.sug_params = None
        self.final_params = None
        self.good_rectify_flag_list = None
        # self.new_params_reserved_dict = None
        # rectify:
        # get: self: old_params&new_params
        # return: None (change new_params)
        # if new_params is None return False

        self.json = None  # search space
        self.total_data = {}  # dict: id -> parameter(_index _value)
        self.rval = None
        self.rval_rectify = None  # none parallel
        self.supplement_data_num = 0
        self.rectify_probability = None

        self.parallel = parallel
        self.parallel_optimize = self.parallel["parallel_optimize"]
        if self.parallel_optimize:
            self.CL_rval = None
            self.constant_liar_type = self.parallel["constant_liar_type"]
            self.running_data = []
            self.optimal_y = None

        # 格式问题：存在三种不同的参数字典格式 1 2 3
        # e.g.
        # {'root[act_func]-choice': 1}
        # {'bn_layer': {'_index': 1, '_value': 'false'}}
        # {'gamma': 0.27984980742661936}
        # misc ...
        # search space instance ...

    def complete_config_by_default(self):
        pass

    def get_exp_and_id(self):
        # trial_env_vars = _load_env_vars(_trial_env_var_names)
        dispatcher_env_vars = _load_env_vars(_dispatcher_env_var_names)
        platform_log_dir = str(dispatcher_env_vars.NNI_LOG_DIRECTORY)
        exp_id = platform_log_dir.split("/")[-2]

        experiment_list_path = Path.home() / 'nni-experiments/.experiment'
        experiments = json.load(open(experiment_list_path))
        port = [exp['port'] for exp in experiments.values() if exp['id'] == exp_id][0]
        self.exp = Experiment(None).connect(port)
        self.exp_id = exp_id
        logger.info(" ".join(["exp_id:", self.exp_id]))

    def get_exec_duration(self):
        # metadata = exp.get_experiment_metadata(self.exp_id)
        profile = self.exp.get_experiment_profile()
        # logger.info("profile: " + str(profile) + "\n")
        d = profile["execDuration"]
        return d

    def init_max_duration(self):
        dispatcher_env_vars = _load_env_vars(_dispatcher_env_var_names)
        _dir = "/".join(dispatcher_env_vars.NNI_LOG_DIRECTORY.split("/")[:-2])
        logger.info("dir: " + str(_dir))
        config = Config(self.exp_id, _dir).get_config()
        self.max_duration = parse_time(config["maxExperimentDuration"])
        logger.info(" ".join(["max_duration:", str(self.max_duration)]))

    def init_duration_probability_list(self):
        max_dur = self.max_duration

        # y = kx + b (k>0)
        d_list = []
        p_list = []
        start_d = int(round(self.start_duration_float * max_dur))
        end_d = int(round(self.end_duration_float * max_dur))
        # [0+10,10+5,15+3,18+2,20+1]
        # 20-[1,2,3,5,10]

        dur = start_d
        while True:
            if dur > self.end_duration_float * max_dur - 60 or dur in d_list:
                break
            d_list.insert(0, dur)
            p_list.insert(0, min(self.max_p, dur / max_dur * self.k))
            dur = dur + int(np.floor((end_d - dur) / 2))  # not ceil
        d_list.reverse()
        p_list.reverse()

        logger.info("duration_list: " + str(d_list))
        logger.info("probability_list: " + str(p_list))
        # [1, 2, 3, 5, 10]
        # [5.0, 10.0, 15.0, 25.0, 50.0]
        # r(20 - [1, 2, 3, 5, 10]) -> (10 15 17 18 19)
        # r((100 - [5.0, 10.0, 15.0, 25.0, 50.0])/100)

        self.duration_list = d_list
        self.probability_list = p_list

    def _choose_tuner(self, algorithm_name):
        if algorithm_name == 'tpe':
            return hp.tpe.suggest
        if algorithm_name == 'random':
            return hp.rand.suggest
        if algorithm_name == 'anneal':
            return hp.anneal.suggest
        raise RuntimeError('Not support tuner algorithm in hyperopt.')

    def update_search_space(self, search_space):
        validate_search_space(search_space)
        self.json = search_space

        search_space_instance = json2space(self.json)
        rstate = np.random.RandomState()
        trials = hp.Trials()
        domain = hp.Domain(None,
                           search_space_instance,
                           pass_expr_memo_ctrl=None)
        algorithm = self._choose_tuner(self.algorithm_name)
        self.rval = hp.FMinIter(algorithm,
                                domain,
                                trials,
                                max_evals=-1,
                                rstate=rstate,
                                verbose=0)
        self.rval_rectify = hp.FMinIter(hp.anneal.suggest,
                                        domain,
                                        trials,
                                        max_evals=-1,
                                        rstate=rstate,
                                        verbose=0)

        self.rval.catch_eval_exceptions = False
        self.rval_rectify.catch_eval_exceptions = False

    def expect_different(self, key, default=True):
        if key in self.old_params:
            if self.old_params[key] != self.sug_params[key]:
                flag = True
                self.final_params.update({key: self.sug_params[key]})
            else:
                flag = False
        else:
            flag = default
        self.good_rectify_flag_list.append(flag)

    def expect_new_true(self, key, default=True):
        if key in self.sug_params:
            if self.sug_params[key] is True:
                flag = True
                self.final_params.update({key: self.sug_params[key]})
            else:
                flag = False
        else:
            flag = default
        self.good_rectify_flag_list.append(flag)

    def expect_new_larger(self, key, default=True):
        if key in self.sug_params:
            if self.sug_params[key] > self.old_params[key]:
                flag = True
                self.final_params.update({key: self.sug_params[key]})
            else:
                flag = False
        else:
            flag = default
        self.good_rectify_flag_list.append(flag)

    def expect_new_smaller(self, key, default=True):
        if key in self.sug_params:
            if self.sug_params[key] > self.old_params[key]:
                flag = True
                self.final_params.update({key: self.sug_params[key]})
            else:
                flag = False
        else:
            flag = default
        self.good_rectify_flag_list.append(flag)

    def rectify_param_follow_at(self):
        if self.at_symptom == "VG":
            self.expect_new_true("bn_layer")
            self.expect_different("act_func")
        if self.at_symptom == "EG":
            self.expect_new_true("bn_layer")
            self.expect_different("act_func")
            self.expect_new_true("grad_clip")
        if self.at_symptom == "DR":
            self.expect_new_true("bn_layer")
            self.expect_different("act_func")
            self.expect_different("init")
        if self.at_symptom == "SC":
            self.expect_different("init")
            self.expect_different("opt")
            self.expect_new_larger("lr")
        if self.at_symptom == "OL":
            self.expect_different("init")
            self.expect_different("opt")
            self.expect_new_larger("batch_size")
            self.expect_new_smaller("lr")

    def rectify_param_follow_dd(self):
        if self.dd_symptom == "ExplodingTensor":  # act lr init # cond: weight data learn
            if self.weight_cond is False:
                self.expect_different("init")  # 4
            elif self.data_cond is False:
                self.good_rectify_flag_list.append(True)  # 0 # improper data
            elif self.lr_cond is False:  # cur low lr
                self.expect_new_smaller("lr")  # 3
            else:
                self.expect_different("act_func")  # 2
        if self.dd_symptom == "UnchangedWeight":
            if self.weight_cond is False:
                self.expect_different("init")  # 4
            elif self.lr_cond is False:
                self.expect_different("act_func")  # 3
            else:
                self.expect_different("opt")  # 6
        if self.dd_symptom == "LossNotDecreasing" or self.dd_symptom == "AccuracyNotIncreasing":
            if self.weight_cond is False:
                self.expect_different("init")  # 4
            elif self.data_cond is False:
                self.good_rectify_flag_list.append(True)  # 0 # improper data
            elif self.lr_cond is False:
                self.expect_new_larger("lr")  # 3
            else:
                self.expect_different("opt")  # 6
        if self.dd_symptom == "VanishingGradient":
            if self.lr_cond is False:
                self.expect_new_larger("lr")  # 3
            else:
                self.rectify_params_healthy()
                # self.good_rectify_flag_list.append(True)  # 5 # change layer number
            pass

    def rectify_param_follow_wd(self):
        if self.wd_symptom is not None:
            if self.weight_cond is False:
                self.expect_different("init")  # 4
            elif self.data_cond is False:
                self.good_rectify_flag_list.append(True)  # 0 # improper data
            elif self.lr_cond is False:
                self.expect_new_larger("lr")  # 3
            else:
                self.expect_different("opt")  # 6

    def rectify_params_healthy(self):
        self.expect_new_larger("epoch")
        self.expect_different("seed")
        if_rectify_flag = False

        mid = int(np.ceil(self.same_retry_maximum / 2))
        for i in range(self.same_retry_maximum - 1):
            for key in self.sug_params:
                if key not in self.retained_parameter_name_list:  # 约后期改得越少
                    r = random.random()
                    if (i <= mid and r > self.rectify_probability) or (i > mid and r > self.rectify_probability):
                        self.final_params.update({key: self.sug_params[key]})
                        self.good_rectify_flag_list.append(True)
                        if_rectify_flag = True
            if if_rectify_flag is True and self.final_params not in self.id_parameters_dict_dict.values():
                # e.g. bn_layer期望为True但是相当于没改！
                return

        for key in self.sug_params:
            self.final_params.update({key: self.sug_params[key]})
            self.good_rectify_flag_list.append(True)

    def rectify_parameters(self):
        good_flag = True
        self.final_params = self.old_params.copy()  ######

        d = {
            "at": self.rectify_param_follow_at,
            "dd": self.rectify_param_follow_dd,
            "wd": self.rectify_param_follow_wd,
        }
        for key, func in d.items():
            if key in self.rule_name_list:
                if self.optimal_dict["result_dict"][key + "_symptom"] is not None:
                    self.good_rectify_flag_list = []
                    func()
                    if False in self.good_rectify_flag_list:  # 紧
                        logger.info(" ".join([key, "bad rectify!"]))
                        return False
                    logger.info(" ".join([key, "good rectify!"]))
        if self.at_symptom is None and self.dd_symptom is None and self.wd_symptom is None \
                or self.final_params in self.id_parameters_dict_dict.values():
            self.good_rectify_flag_list = []
            self.rectify_params_healthy()
            if False in self.good_rectify_flag_list:  # 紧
                logger.info(" ".join(["health", "bad rectify!"]))
                return False
            logger.info(" ".join(["health", "good rectify!"]))

        if self.final_params in self.id_parameters_dict_dict.values():
            logger.info(" ".join(["same param even after health rectify! :", str(self.final_params)]))
            return False  # 不重复！ 重复参数无意义。。。 # 应该和s
        return True  # good flag

    def update_rectify_probability(self):
        p = None
        d = self.get_exec_duration()
        md = self.max_duration
        if len(self.duration_list) == 0 or self.rectify_probability is None:
            p = self.base_p  # no
        else:
            for i in range(len(self.duration_list)):
                if d <= self.duration_list[i]:
                    p = self.probability_list[i - 1] if i != 0 else self.base_p
                    break
            if p is None:
                p = 1
        if self.rectify_probability is None or p != self.rectify_probability:
            s = " ".join(["update rectify probability:", str(p), "execDuration:", str(d), "maxDuration:", str(md)])
            logger.info(s)
        self.rectify_probability = p

    def generate_parameters(self, parameter_id, **kwargs):
        logger.info(" ".join(["begin gen params:", "parameter_id:", parameter_id]))  ###
        self.update_rectify_probability()
        # logger.info(" ".join(["rectify_probability:", str(self.rectify_probability), "\n"]))
        total_params = self._get_suggestion(random_search=False, self_rval=self.rval)
        if self.old_params is None or random.random() > self.rectify_probability:
            pass  # 不修复
        else:
            s = " ".join([str(self.at_symptom), str(self.dd_symptom), str(self.wd_symptom)])
            # logger.info(" ".join(["optimal trial_id:", self.optimal_dict["result_dict"]["trial_id"]]))
            logger.info(" ".join(["optimal trial_id:", self.optimal_dict["trial_id"]]))
            logger.info(" ".join(["rectify symptom: at dd wd :", s]))
            # logger.info(" ".join(["pre param:", str(self.old_params)]))
            # logger.info(" ".join(["sug param:", str(self.sug_params)]))
            good_flag = None
            for i in range(self.same_retry_maximum):
                # 修复判断 存在 index 格式问题 ！！！！！！！！！！！！
                total_params = self._get_suggestion(random_search=True, self_rval=self.rval_rectify)
                self.sug_params = split_index(total_params)
                good_flag = self.rectify_parameters()
                if good_flag:
                    # logger.info(" ".join(["final param:", str(self.final_params)]))
                    total_params = _add_index(self.json, self.final_params)
                    break
            if good_flag is False:
                logger.info(" ".join(["exceed same_retry_maximum:", str(self.same_retry_maximum)]))

        self.total_data[parameter_id] = total_params

        if self.parallel_optimize:
            self.running_data.append(parameter_id)

        params = split_index(total_params)
        # logger.info("\n".join(["split_index: ", "before: " + str(total_params), "after: " + str(params), "\n"]))
        self.id_parameters_dict_dict[parameter_id] = params
        return params

    def update_optimal(self, parameter_id, trial_id, result_dict):
        update_flag = False
        if self.optimal_dict is None:
            update_flag = True
        else:
            val = result_dict["default"]
            opt_val = self.optimal_dict["result_dict"]["default"]
            if self.optimize_mode == "maximize" and val > opt_val:
                update_flag = True
            if self.optimize_mode == "minimize" and val < opt_val:
                update_flag = True

        if update_flag is True:
            d = {}
            d.update({"trial_id": trial_id})
            d.update({"result_dict": result_dict})
            d.update({"parameters_dict": self.id_parameters_dict_dict[parameter_id]})
            self.optimal_dict = d
            self.old_params = self.id_parameters_dict_dict[parameter_id]
            self.at_symptom = result_dict["at_symptom"] if "at_symptom" in result_dict.keys() else None
            self.dd_symptom = result_dict["dd_symptom"] if "dd_symptom" in result_dict.keys() else None
            self.wd_symptom = result_dict["wd_symptom"] if "wd_symptom" in result_dict.keys() else None
            self.data_cond = result_dict["data_cond"]
            self.weight_cond = result_dict["weight_cond"]
            self.lr_cond = result_dict["lr_cond"]
            logger.info(" ".join(["update optimal:", trial_id, str(result_dict["default"])]))
            logger.info(" ".join(["symptom: (at dd wd)",
                                  str(self.at_symptom), str(self.dd_symptom), str(self.wd_symptom)]))
            logger.info(" ".join(["condition (d w l):",
                                  str(self.data_cond), str(self.weight_cond), str(self.lr_cond)]))

    def receive_trial_result(self, parameter_id, parameters, value: Union[dict, float], **kwargs):
        trial_id = kwargs["trial_job_id"]
        logger.info("send final_result_dict: %s: %s" % (trial_id, value["step_counter"]))
        logger.info("stop_signal: (assess inspect) %s: %s %s" % \
                    (trial_id, value["assessor_stop"], value["inspector_stop"]))
        logger.debug("final_result_dict: %s: %s" % (trial_id, str(value)))
        self.update_optimal(parameter_id, trial_id, value)

        reward = extract_scalar_reward(value)
        # restore the paramsters contains '_index'
        if parameter_id not in self.total_data:
            raise RuntimeError('Received parameter_id not in total_data.')
        params = self.total_data[parameter_id]

        for self_rval in [self.rval, self.rval_rectify]:
            # code for parallel
            if self.parallel_optimize:
                constant_liar = kwargs.get('constant_liar', False)

                if constant_liar:
                    rval = self.CL_rval
                else:
                    rval = self.rval
                    # ignore duplicated reported final result (due to aware of intermedate result)
                    if parameter_id not in self.running_data:
                        logger.info("Received duplicated final result with parameter id: %s", parameter_id)
                        return
                    self.running_data.remove(parameter_id)

                    # update the reward of optimal_y
                    if self.optimal_y is None:
                        if self.constant_liar_type == 'mean':
                            self.optimal_y = [reward, 1]
                        else:
                            self.optimal_y = reward
                    else:
                        if self.constant_liar_type == 'mean':
                            _sum = self.optimal_y[0] + reward
                            _number = self.optimal_y[1] + 1
                            self.optimal_y = [_sum, _number]
                        elif self.constant_liar_type == 'min':
                            self.optimal_y = min(self.optimal_y, reward)
                        elif self.constant_liar_type == 'max':
                            self.optimal_y = max(self.optimal_y, reward)
                    logger.debug("Update optimal_y with reward, optimal_y = %s", self.optimal_y)
            else:
                # rval = self.rval
                rval = self_rval
            if self.optimize_mode is OptimizeMode.Maximize:
                reward = -reward

            domain = rval.domain
            trials = rval.trials

            new_id = len(trials)

            rval_specs = [None]
            rval_results = [domain.new_result()]
            rval_miscs = [dict(tid=new_id, cmd=domain.cmd, workdir=domain.workdir)]

            vals = params
            idxs = dict()

            out_y = dict()
            json2vals(self.json, vals, out_y)
            vals = out_y
            for key in domain.params:
                if key in [NodeType.VALUE, NodeType.INDEX]:
                    continue
                if key not in vals or vals[key] is None or vals[key] == []:
                    idxs[key] = vals[key] = []
                else:
                    idxs[key] = [new_id]
                    vals[key] = [vals[key]]

            self._miscs_update_idxs_vals(rval_miscs,
                                         idxs,
                                         vals,
                                         idxs_map={new_id: new_id},
                                         assert_all_vals_used=False)

            trial = trials.new_trial_docs([new_id], rval_specs, rval_results,
                                          rval_miscs)[0]
            trial['result'] = {'loss': reward, 'status': 'ok'}
            trial['state'] = hp.JOB_STATE_DONE
            trials.insert_trial_docs([trial])
            trials.refresh()

    def _miscs_update_idxs_vals(self,
                                miscs,
                                idxs,
                                vals,
                                assert_all_vals_used=True,
                                idxs_map=None):
        """
        Unpack the idxs-vals format into the list of dictionaries that is
        `misc`.

        Parameters
        ----------
        idxs_map : dict
            idxs_map is a dictionary of id->id mappings so that the misc['idxs'] can
        contain different numbers than the idxs argument.
        """
        if idxs_map is None:
            idxs_map = {}

        assert set(idxs.keys()) == set(vals.keys())

        misc_by_id = {m['tid']: m for m in miscs}
        for m in miscs:
            m['idxs'] = {key: [] for key in idxs}
            m['vals'] = {key: [] for key in idxs}

        for key in idxs:
            assert len(idxs[key]) == len(vals[key])
            for tid, val in zip(idxs[key], vals[key]):
                tid = idxs_map.get(tid, tid)
                if assert_all_vals_used or tid in misc_by_id:
                    misc_by_id[tid]['idxs'][key] = [tid]
                    misc_by_id[tid]['vals'][key] = [val]

    def _get_suggestion(self, random_search=False, self_rval=None):
        if self.parallel_optimize and len(self.total_data) > 20 and self.running_data and self.optimal_y is not None:
            self.CL_rval = copy.deepcopy(self.rval)
            if self.constant_liar_type == 'mean':
                _constant_liar_y = self.optimal_y[0] / self.optimal_y[1]
            else:
                _constant_liar_y = self.optimal_y
            for _parameter_id in self.running_data:
                self.receive_trial_result(parameter_id=_parameter_id, parameters=None, value=_constant_liar_y,
                                          constant_liar=True)
            rval = self.CL_rval

            random_state = np.random.randint(2 ** 31 - 1)
        else:
            rval = self_rval if self_rval is not None else self.rval
            random_state = rval.rstate.randint(2 ** 31 - 1)

        trials = rval.trials
        algorithm = rval.algo
        new_ids = rval.trials.new_trial_ids(1)
        rval.trials.refresh()

        if random_search:
            new_trials = hp.rand.suggest(new_ids, rval.domain, trials, random_state)
        else:
            new_trials = algorithm(new_ids, rval.domain, trials, random_state)

        rval.trials.refresh()
        vals = new_trials[0]['misc']['vals']
        parameter = dict()
        for key in vals:
            try:
                parameter[key] = vals[key][0].item()
            except (KeyError, IndexError):
                parameter[key] = None

        # remove '_index' from json2parameter and save params-id
        total_params = json2parameter(self.json, parameter)
        # logger.warn("\n".join(["json2parameter: ", "before: " + str(parameter), "after: " + str(total_params), "\n"]))
        return total_params

    def import_data(self, data):
        # for resume
        _completed_num = 0
        for trial_info in data:
            logger.info("Importing data, current processing progress %s / %s", _completed_num, len(data))
            _completed_num += 1
            if self.algorithm_name == 'random_search':
                return
            assert "parameter" in trial_info
            _params = trial_info["parameter"]
            assert "value" in trial_info
            _value = trial_info['value']
            if not _value:
                logger.info("Useless trial data, value is %s, skip this trial data.", _value)
                continue
            self.supplement_data_num += 1
            _parameter_id = '_'.join(
                ["ImportData", str(self.supplement_data_num)])
            self.total_data[_parameter_id] = _add_index(in_x=self.json,
                                                        parameter=_params)
            self.receive_trial_result(parameter_id=_parameter_id,
                                      parameters=_params,
                                      value=_value)
        logger.info("Successfully import data to TPE/Anneal tuner.")
