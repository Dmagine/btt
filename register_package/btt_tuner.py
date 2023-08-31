import importlib
import logging
import sys

logger = logging.getLogger(__name__)


class BttTuner:
    hpo_config = None
    hpo_name_list = None
    hpo_instance_dict = None
    default_hpo_name = None

    # generate_params()
    # receive_final_result()
    # 	default

    def __init__(self, hpo_config):
        logger.debug("monitor hello!")
        # 组合策略 多半是前期random 后期bo
        self.hpo_config = hpo_config  # hpo_name init_args start_time_ratio end_time_ratio
        self.hpo_name_list = list(self.hpo_config.keys())
        self.init_hpo_instance()

    def init_hpo_instance(self):
        for hpo_name, hpo_info in self.hpo_config.items():
            code_dir = hpo_info["code_dir"]
            module_name = hpo_info["module_name"]
            class_name = hpo_info["class_name"]
            init_args = hpo_info["init_args"] if "init_args" in hpo_info else {}
            init_args["hpo_name"] = hpo_name  ###

            sys.path.append(code_dir)
            class_module = importlib.import_module(module_name)
            class_constructor = getattr(class_module, class_name)
            instance = class_constructor(init_args)
            self.hpo_instance_dict[hpo_name] = instance

            if "default" in hpo_info and hpo_info["default"]:
                self.default_hpo_name = hpo_name

    def obtain_trial_parameters(self):
        def get_experiment_time_ratio():
            # .... nni bala
            return 0

        for hpo_instance in self.hpo_instance_dict.values():
            s_time_ratio = hpo_instance.start_ratio
            e_time_ratio = hpo_instance.end_time_ratio
            if s_time_ratio <= get_experiment_time_ratio() <= e_time_ratio:
                return hpo_instance.obtain_trial_parameters()
        return self.hpo_instance_dict[self.default_hpo_name].obtain_trial_parameters()

    def report_trial_final_result(self, result_dict):
        for hpo_instance in self.hpo_instance_dict.values():
            hpo_instance.report_trial_final_result(result_dict)
        return
