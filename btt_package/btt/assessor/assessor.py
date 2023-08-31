import logging
import threading

from ..experiment import TrialStatus, SharedExpData, ExperimentConfig  # package内部用相对import

#


logger = logging.getLogger(__name__)


class Assessor:
    def __init__(self, exp_config: ExperimentConfig, exp_data: SharedExpData):
        self.logger = logging.getLogger(self.__class__.__name__)
        logger.debug("{} init".format(self.logger.name))
        self.exp_config = exp_config  # hpo_name init_args start_time_ratio end_time_ratio
        self.exp_data = exp_data

        self.indicator_instance_dict = None
        self.heartbeat_interval = 1  # 1s
        self.heartbeat_thread = None

        self.id_idx_dict = None  # idx -> last_assessed_interm_result_idx

    def start(self):
        self.init_indicator_instance_list()
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_thread)
        self.heartbeat_thread.start()

    def init_indicator_instance_list(self):
        self.indicator_instance_dict = {}
        for indicator_name, indicator_config in self.exp_config.assessor_config.items():
            indicator_instance = indicator_config.class_ctor(indicator_config.init_args)
            self.indicator_instance_dict[indicator_name] = indicator_instance

    def stop(self):
        self.heartbeat_thread.join()

    def _heartbeat_thread(self):
        id_len_dict = {}  # idx -> interm_result_len when last_assessed -> 1,2,3...
        while True:
            if self.exp_data.get_exp_stop_flag():
                return
            # epoch_quota
            for trial_id, trial_data in self.exp_data.items():
                trial_data.lock.acquire()
                # initial_result
                if trial_data.status != TrialStatus.Waiting and trial_data.initial_result is not None \
                        and trial_data.interm_result_list is not None:
                    if trial_data.assessor_result_list is None:
                        trial_data.assessor_result_list = []
                    idx_start = len(trial_data.interm_result_list)
                    idx_end = len(trial_data.assessor_result_list)
                    if idx_start < idx_end:
                        initial_result = trial_data.initial_result
                        for idx in range(idx_start, idx_end):
                            interm_result = trial_data.interm_result_list[idx]
                            assessor_result = self.assess_interm_result(trial_id, initial_result, interm_result)
                            trial_data.assessor_result_list.append(assessor_result)
                trial_data.lock.release()

    def assess_interm_result(self):
        # 只需要传递增量，不需要传递全部
        d = {}
        for indicator_name, indicator_instance in self.indicator_instance_dict.items():
            d[indicator_name] = indicator_instance.assess_trial(trial_id, initial_dict, intermediate_dict_list)
