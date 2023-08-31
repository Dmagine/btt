import threading
from copy import deepcopy

from btt.exp_manager import ExperimentConfig, TrialStatus
from btt.exp_manager import SharedExpData


class Allocator:
    # allocator 未来有 policy
    def __init__(self, exp_config: ExperimentConfig, exp_data: SharedExpData):
        self.trial_gpu_number = exp_config.trial_gpu_number
        self.available_gpu_idx_list = exp_config.available_gpu_idx_list
        self.exp_data = exp_data

        self.stop_flag = False
        self.heartbeat_interval = 1  # 1s
        self.heartbeat_thread = None

    def start(self):
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_thread)
        self.heartbeat_thread.start()

    def stop(self):
        self.stop_flag = True
        self.heartbeat_thread.join()

    def _heartbeat_thread(self):
        # 只有一个allocator在控制device
        while True:
            # device
            if self.stop_flag:
                return
            gpu_idx_list = deepcopy(self.available_gpu_idx_list)
            for trial_id, trial_data in self.exp_data.items():
                if trial_data.status == TrialStatus.Running and trial_data.device_str != "cpu":
                    gpu_idx_list.remove(int(trial_data.device_str.split(":")[1]))
            for trial_id, trial_data in self.exp_data.items():
                trial_data.lock.acquire()
                if trial_data.status == TrialStatus.Waiting and trial_data.device_str is None:
                    if self.trial_gpu_number == 0:
                        trial_data.device_str = "cpu"
                    elif self.trial_gpu_number == 1:
                        trial_data.device_str = "cuda:" + str(gpu_idx_list[0])
                        gpu_idx_list.pop(0)
                trial_data.lock.release()
            time.sleep(self.heartbeat_interval)
