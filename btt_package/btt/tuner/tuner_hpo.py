import logging
from abc import abstractmethod

from nni.tools.package_utils import create_builtin_class_instance


class TunerHpoBase:
    def __init__(self, d_args):
        self.hpo_name = d_args["hpo_name"]
        self.start_time_ratio = d_args["start_time_ratio"] if "start_time_ratio" in d_args else 0
        self.end_time_ratio = d_args["end_time_ratio"] if "end_time_ratio" in d_args else 1
        self.logger = logging.getLogger(self.hpo_name)
        # self.logger.setLevel(logging.DEBUG)
        pass

    @abstractmethod
    def obtain_trial_parameters(self):
        raise NotImplementedError

    @abstractmethod
    def report_trial_final_result(self, d_args):
        raise NotImplementedError


class NniTunerHpo(TunerHpoBase):

    def __init__(self, d_args):
        super().__init__(d_args)
        self.logger.debug("NniTunerHpo init")
        self.nni_tuner = create_builtin_class_instance(d_args["hpo_name"], d_args["init_args"], "tuners")

    def obtain_trial_parameters(self):
        #     def generate_parameters(self, parameter_id, **kwargs):
        return self.nni_tuner.generate_parameters()  # 可以直接return吗？

    def report_trial_final_result(self, d_args):
        #     def receive_trial_result(self, parameter_id, parameters, value, **kwargs):
        self.nni_tuner.receive_trial_result(d_args)
        pass
