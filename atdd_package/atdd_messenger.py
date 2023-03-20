import os
import sys

import yaml
from nni.common.serializer import dump, load
from nni.runtime.env_vars import _load_env_vars, _trial_env_var_names, _dispatcher_env_var_names

info_file_name_dict = {
    'monitor': 'monitor_info',
    'assessor': 'assessor_info',
    'inspector': 'inspector_info',
    'advisor_config': 'advisor_config_info',
    'tuner': 'tuner_info',
    'other': 'other_info'
}


class ATDDMessenger:
    def __init__(self, trial_id=None):
        self.trial_id = trial_id
        self.trial_nni_dir = None
        self.platform_trials_dir = None
        # self.step_counter = 0

    def get_file_path(self, key):
        file_path = None
        # file_path = None
        # if key in self.trial_info_path_dict.keys():
        #     file_path = self.trial_info_path_dict[key] if idx is None \
        #         else "_".join([self.trial_info_path_dict[key], str(idx)])
        # if key in self.experiment_info_path_dict.keys():
        #     file_path = self.experiment_info_path_dict[key]

        if key == "monitor":  # monitor write / inspector read
            trial_env_vars = _load_env_vars(_trial_env_var_names)
            trial_system_dir = trial_env_vars.NNI_SYS_DIR
            self.trial_nni_dir = os.path.join(trial_system_dir, '.nni')
            if not os.path.exists(self.trial_nni_dir):
                os.makedirs(self.trial_nni_dir)
            file_path = os.path.join(self.trial_nni_dir, info_file_name_dict[key])
        if key == "assessor":  # assessor write / manager read
            if get_nni_context() == "platform":
                dispatcher_env_vars = _load_env_vars(_dispatcher_env_var_names)
                platform_log_dir = dispatcher_env_vars.NNI_LOG_DIRECTORY
                trial_system_dir = os.path.join(platform_log_dir, '../trials', self.trial_id)
            else:
                trial_env_vars = _load_env_vars(_trial_env_var_names)
                trial_system_dir = trial_env_vars.NNI_SYS_DIR
            self.trial_nni_dir = os.path.join(trial_system_dir, '.nni')
            if not os.path.exists(self.trial_nni_dir):
                os.makedirs(self.trial_nni_dir)
            file_path = os.path.join(self.trial_nni_dir, info_file_name_dict[key])
        if key == "inspector":  # inspector write / manager read
            trial_env_vars = _load_env_vars(_trial_env_var_names)
            trial_system_dir = trial_env_vars.NNI_SYS_DIR
            self.trial_nni_dir = os.path.join(trial_system_dir, '.nni')
            if not os.path.exists(self.trial_nni_dir):
                os.makedirs(self.trial_nni_dir)
            file_path = os.path.join(self.trial_nni_dir, info_file_name_dict[key])
        if key == "advisor_config":  # advisor write / monitor read
            # special
            if get_nni_context() == "platform":
                dispatcher_env_vars = _load_env_vars(_dispatcher_env_var_names)
                platform_log_dir = dispatcher_env_vars.NNI_LOG_DIRECTORY
                self.platform_trials_dir = os.path.join(platform_log_dir, '../trials/')
            else:
                trial_env_vars = _load_env_vars(_trial_env_var_names)
                trial_system_dir = trial_env_vars.NNI_SYS_DIR
                self.platform_trials_dir = os.path.join(trial_system_dir, '../')
            if not os.path.exists(self.platform_trials_dir):
                os.makedirs(self.platform_trials_dir)
            file_path = os.path.join(self.platform_trials_dir, info_file_name_dict[key])
        if key == "default_config":
            d = "./"
            for p in sys.path:
                if "atdd_package" in p:
                    d = p
                    break
            file_path = os.path.join(d, "atdd_default_all_soft.yaml")  # .../atdd_package/default_soft.yaml
        if key == "tuner":
            dispatcher_env_vars = _load_env_vars(_dispatcher_env_var_names)
            platform_log_dir = dispatcher_env_vars.NNI_LOG_DIRECTORY
            self.platform_trials_dir = os.path.join(platform_log_dir, '../trials/')
            if not os.path.exists(self.platform_trials_dir):
                os.makedirs(self.platform_trials_dir)
            file_path = os.path.join(self.platform_trials_dir, info_file_name_dict[key])
        return file_path

    def write_json_info(self, info_dict, key):
        dumped_info = dump(info_dict)
        data = (dumped_info + '\n').encode('utf8')
        assert len(data) < 1000000, 'Info too long'

        file_path = self.get_file_path(key)
        file_obj = open(file_path, 'wb')
        file_obj.write(data)
        file_obj.flush()
        file_obj.close()

    def read_json_info(self, key):
        file_path = self.get_file_path(key)
        if os.path.isfile(file_path) and os.path.getsize(file_path) > 0:
            file_obj = open(file_path, 'r')
            info_dict = load(fp=file_obj)
            file_obj.close()
            return info_dict
        return None

    def read_yaml_info(self, key):
        file_path = self.get_file_path(key)
        if os.path.isfile(file_path) and os.path.getsize(file_path) > 0:
            file_obj = open(file_path, 'r')
            info_dict = yaml.load(file_obj, Loader=yaml.FullLoader)
            file_obj.close()
            return info_dict
        return None

    def write_assessor_info(self, d):
        self.write_json_info(d, key='assessor')

    def read_assessor_info(self):
        return self.read_json_info(key='assessor')

    def write_monitor_info(self, d):
        self.write_json_info(d, key='monitor')

    def read_monitor_info(self):
        return self.read_json_info(key='monitor')

    def write_inspector_info(self, d):
        self.write_json_info(d, key='inspector')

    def read_inspector_info(self):  # for tuner
        return self.read_json_info(key='inspector')

    def write_advisor_config(self, d):
        self.write_json_info(d, key='advisor_config')

    def read_advisor_config(self):  # for tuner
        return self.read_json_info(key='advisor_config')

    def read_default_config_info(self):
        return self.read_yaml_info(key='default_config')

    def read_tuner_info(self):
        return self.read_json_info(key='tuner')

    def write_tuner_info(self, d):
        self.write_json_info(d, key='tuner')


def get_nni_context():
    e1, e2 = os.environ.get('NNI_EXP_ID') is None, os.environ.get('NNI_LOG_DIRECTORY') is None
    if e1 is True and e2 is True:
        return "raw"
    elif e1 is True and e2 is False:
        return "platform"
    elif e1 is False and e2 is True:
        return "trial"
    else:
        raise Exception("mode????")


def if_enable(lst):
    enable_dict = ATDDMessenger().read_advisor_config()["shared"]["enable_dict"]
    for name in lst:
        if name not in enable_dict:
            return False
        if enable_dict[name] is False:
            return False
    return True
