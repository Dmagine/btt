import logging
from abc import abstractmethod


class TunerHpoBase:
    def __init__(self, hpo_name):
        self.hpo_name = hpo_name
        self.logger = logging.getLogger(hpo_name)
        # self.logger.setLevel(logging.DEBUG)
        pass

    @abstractmethod
    def generate_params(self, trial_id):
        raise NotImplementedError

    @abstractmethod
    def receive_final_result(self, trial_id, final_result):
        raise NotImplementedError


class RandomHpo(TunerHpoBase):
    def __init__(self, hpo_name):
        super().__init__(hpo_name)
        pass

    def generate_params(self, trial_id):
        pass

    def receive_final_result(self, trial_id, final_result):
        pass
