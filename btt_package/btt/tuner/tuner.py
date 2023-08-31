import logging
import threading
import time

from ..experiment import TrialStatus, SharedExpData, ExperimentConfig  # package内部用相对import

#


logger = logging.getLogger(__name__)


class Tuner:
    def __init__(self, exp_config: ExperimentConfig, exp_data: SharedExpData):
        self.logger = logging.getLogger(self.__class__.__name__)
        logger.debug("{} init".format(self.logger.name))
        self.exp_config = exp_config  # hpo_name init_args start_time_ratio end_time_ratio
        self.exp_data = exp_data

        self.hpo_instance_dict = None
        self.heartbeat_interval = 1  # 1s
        self.heartbeat_thread = None

    def start(self):
        self.init_hpo_instance_list()
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_thread)
        self.heartbeat_thread.start()

    def stop(self):
        self.heartbeat_thread.join()

    def _heartbeat_thread(self):
        while True:
            # device
            if self.exp_data.get_exp_stop_flag():
                return
            for trial_id, trial_data in self.exp_data.items():
                # params
                trial_data.lock.acquire()
                if trial_data.status == TrialStatus.Waiting and trial_data.params is None:
                    trial_data.params = self.generate_params(trial_id)  # 非最后真param 但是可以避免"相同"trial出现
                if trial_data.status != TrialStatus.Waiting and trial_data.final_result is not None:
                    self.receive_final_result(trial_id, trial_data.final_result)
                trial_data.lock.release()

    def init_hpo_instance_list(self):
        for hpo_name, hpo_config in self.exp_config.tuner_config.items():
            hpo_instance = hpo_config.class_ctor(hpo_config.init_args)
            self.hpo_instance_dict[hpo_name] = hpo_instance

    def generate_params(self, trial_id):
        for hpo_name, hpo_instance in self.hpo_instance_dict.items():
            sr = hpo_instance.exp_config.start_ratio
            er = hpo_instance.exp_config.end_ratio
            # 需要知道exp_config的一些内容辅助计算
            if self.exp_config.tuner_config[hpo_name].dur_not_num is True:
                tmp_r = (time.time() - self.exp_data.get_exp_start_time()) / self.exp_config.max_exp_duration
            else:
                tmp_r = (self.exp_config.max_trial_number - len(self.exp_data)) / self.exp_config.max_trial_number
            if sr <= tmp_r <= er:
                return hpo_instance.generate_params()
        return

    def receive_final_result(self, trial_id, final_result):
        for hpo_instance in self.hpo_instance_dict.values():
            hpo_instance.receive_final_result(trial_id, final_result)
        return
