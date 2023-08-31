import logging
import threading

from .experiment import ExperimentConfig  # package内部用相对import


class Trial:
    def __init__(self, trial_id, trial_data, exp_config: ExperimentConfig, seed=529):
        self.logger = logging.getLogger(self.__class__.__name__)
        # set_seed(seed, "btt_trial_manager", logger)
        self.trial_id = trial_id
        self.trial_data = trial_data
        self.exp_config = exp_config

        # 注：capture和obtain不通过cmd_queue，只有report才调用 （monitor还是在本地侧）
        self.heartbeat_thread = None
        self.heartbeat_interval = 1  # 1s

    def start(self):
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_thread)
        self.heartbeat_thread.start()

    def _heartbeat_thread(self):
        pass

    def capture(self, d_args):
        # record_metric
        pass

    def obtain(self, d_args):
        # obtain_metric
        pass

    def suggest(self):
        # get_trial_parameters
        pass

    # def get_experiment_id(self):
    #     pass
    #
    # def get_trial_id(self):
    #     pass
    #
    # def get_trial_parameters(self):
    #     pass

    # def update_resource_params(self, resource_params):
    #     pass # device废弃 中途修改device不合理 不是主线 （一开始就要确定device，比param还早

    def get_device_str(self):
        pass

    def report_intermediate_result(self):
        if self.monitor_config is not None:
            self.monitor.report_intermediate_result()

    def report_final_result(self):
        if self.monitor_config is not None:
            self.monitor.report_final_result()

    def report_initial_result(self):
        if self.monitor_config is not None:
            self.monitor.report_initial_result()
