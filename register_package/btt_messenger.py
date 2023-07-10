import logging
import os
import sys

import nni
from nni.common.serializer import dump, load
from nni.runtime.env_vars import _load_env_vars, _trial_env_var_names, _dispatcher_env_var_names

logger = logging.getLogger('btt_messenger')

info_file_name_dict = {
    'monitor': 'monitor_info',
    # 'assessor': 'assessor_info',
    'advisor': 'advisor_info',
    # 'tuner': 'tuner_info',
}


class BttMessenger:
    def __init__(self, trial_id=None):
        self.trial_id = trial_id
        self.trial_nni_dir = None
        self.platform_trials_dir = None
        # self.step_counter = 0

    def get_file_path(self, file_name_key):
        file_path = None

        path_to_new_trial = "./"
        ctx = get_nni_context()
        if get_nni_context() == "platform":
            dispatcher_env_vars = _load_env_vars(_dispatcher_env_var_names)
            ver = nni.__version__
            if True in [ver.startswith("2.99"), ver.startswith("3.")]:
                path_to_new_trial = "../environments/local-env/trials/"
        elif ctx == "trial":
            trial_env_vars = _load_env_vars(_trial_env_var_names)
            trial_system_dir = trial_env_vars.NNI_SYS_DIR
            if "environments" in trial_system_dir:
                path_to_new_trial = "../environments/local-env/trials/"
        else:
            pass

        if file_name_key == "monitor":  # monitor write / inspector read
            trial_env_vars = _load_env_vars(_trial_env_var_names)
            trial_system_dir = trial_env_vars.NNI_SYS_DIR
            self.trial_nni_dir = os.path.join(trial_system_dir, '.nni')
            if not os.path.exists(self.trial_nni_dir):
                os.makedirs(self.trial_nni_dir)
            file_path = os.path.join(self.trial_nni_dir, info_file_name_dict[file_name_key])
        if file_name_key == "assessor":  # assessor write / manager read
            if get_nni_context() == "platform":
                dispatcher_env_vars = _load_env_vars(_dispatcher_env_var_names)
                platform_log_dir = dispatcher_env_vars.NNI_LOG_DIRECTORY

                trial_system_dir = os.path.join(platform_log_dir, '../trials/', path_to_new_trial, self.trial_id)
            else:
                trial_env_vars = _load_env_vars(_trial_env_var_names)
                trial_system_dir = trial_env_vars.NNI_SYS_DIR
            self.trial_nni_dir = os.path.join(trial_system_dir, '.nni')
            if not os.path.exists(self.trial_nni_dir):
                os.makedirs(self.trial_nni_dir)
            file_path = os.path.join(self.trial_nni_dir, info_file_name_dict[file_name_key])
        if file_name_key == "advisor":  # advisor write / monitor read
            # special
            ctx = get_nni_context()
            if ctx == "platform":
                dispatcher_env_vars = _load_env_vars(_dispatcher_env_var_names)
                platform_log_dir = dispatcher_env_vars.NNI_LOG_DIRECTORY
                self.platform_trials_dir = os.path.join(platform_log_dir, '../trials/', path_to_new_trial)
            elif ctx == "trial":
                trial_env_vars = _load_env_vars(_trial_env_var_names)
                trial_system_dir = trial_env_vars.NNI_SYS_DIR
                self.platform_trials_dir = os.path.join(trial_system_dir, '../')
            else:  # no_nni_manager
                self.platform_trials_dir = "./"
            if not os.path.exists(self.platform_trials_dir):
                os.makedirs(self.platform_trials_dir)
            file_path = os.path.join(self.platform_trials_dir, info_file_name_dict[file_name_key])
        if file_name_key == "default_config":
            d = "./"
            for p in sys.path:
                if "new_package" in p:  ###
                    d = p
                    break
            file_path = os.path.join(d, "atdd_default_config.yaml")
            logger.info("default config file_path: {}".format(file_path))
        if file_name_key == "tuner":
            dispatcher_env_vars = _load_env_vars(_dispatcher_env_var_names)
            platform_log_dir = dispatcher_env_vars.NNI_LOG_DIRECTORY
            self.platform_trials_dir = os.path.join(platform_log_dir, '../trials/', path_to_new_trial)
            if not os.path.exists(self.platform_trials_dir):
                os.makedirs(self.platform_trials_dir)
            file_path = os.path.join(self.platform_trials_dir, info_file_name_dict[file_name_key])
        return file_path

    def write_json_info(self, info_dict, file_name_key):
        d = info_dict
        dumped_info = dump(d)
        data = dumped_info.encode('utf8')
        assert len(data) < 1000000, 'Info too long'

        file_path = self.get_file_path(file_name_key)
        file_obj = open(file_path, 'wb')
        file_obj.write(data)
        file_obj.flush()
        file_obj.close()

    def read_json_info(self, file_name_key):
        file_path = self.get_file_path(file_name_key)
        if os.path.isfile(file_path) and os.path.getsize(file_path) > 0:
            file_obj = open(file_path, 'r')
            info_dict = load(fp=file_obj)
            file_obj.close()
            return info_dict
        return None

    def add_json_info(self, info_key, info_value, file_name_key):
        pre_info = self.read_json_info(file_name_key)
        now_info = pre_info if pre_info is not None else {}
        now_info.update({info_key: info_value})
        logger.info("now_info: {}".format(now_info))
        dumped_now_info = dump(now_info)
        data = dumped_now_info.encode('utf8')
        assert len(data) < 1000000, 'Info too long'

        file_path = self.get_file_path(file_name_key)
        file_obj = open(file_path, 'wb')
        file_obj.write(data)
        file_obj.flush()
        file_obj.close()

    def add_intermediate_monitor_result(self, info_value, idx):
        info_key = "_".join(["intermediate", str(idx)])
        self.add_json_info(info_key, info_value, "monitor")

    def add_final_monitor_result(self, info_value):
        self.add_json_info("final", info_value, "monitor")

    def read_resource_params(self, pre_resource_params):
        assessor_info = self.read_json_info("assessor")
        if assessor_info is None or "resource_params" not in assessor_info or assessor_info["resource_params"] is None:
            return pre_resource_params
        return assessor_info["resource_params"]

    def write_advisor_config(self, info_dict):
        self.write_json_info(info_dict, "advisor")

    def read_advisor_config(self):
        return self.read_json_info("advisor")


def get_nni_context():
    e1, e2 = os.environ.get('NNI_EXP_ID') is None, os.environ.get('NNI_LOG_DIRECTORY') is None
    if e1 is True and e2 is True:
        return "no_nni"
    elif e1 is True and e2 is False:
        return "platform"
    elif e1 is False and e2 is True:
        return "trial"
    else:
        raise Exception("mode????")
