import logging
import os
import sqlite3

import yaml

from atdd_messenger import ATDDMessenger

logger = logging.getLogger(__name__)

from nni.tuner import Tuner


class ATDDReproducer(Tuner):
    def __init__(self, reproduce=None):
        self.replay_id = reproduce["experiment_id"]
        logger.info(" ".join(["replay_id:", self.replay_id]))
        self.db_path = None
        self.params_info_list = None
        self.init_params()
        self.counter = 0

    def init_params(self):
        exp_dir = os.path.join("~/nni-experiments/", self.replay_id)
        db_dir = os.path.join(exp_dir, "db")
        db_path_old = os.path.join(db_dir, "nni.sqlite")
        db_path = os.path.join("./", "nni_old.sqlite")  #######
        os.system(" ".join(["cp", db_path_old, db_path]))
        self.db_path = db_path
        logger.info(" ".join(["db_path:", os.path.abspath(self.db_path)]))

        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        sql = "SELECT data FROM TrialJobEvent WHERE event='WAITING'"
        cur.execute(sql)
        values = cur.fetchall()

        self.params_info_list = [None] * len(values)  # tuple
        for i in range(len(values)):
            d = dict(yaml.load(values[i][0], Loader=yaml.FullLoader))
            self.params_info_list[i] = d["parameters"]

        logger.info(" ".join(["replay num:", str(len(self.params_info_list))]))
        logger.debug(" ".join(["show replay last:", str(self.params_info_list[-1])]))

    def update_search_space(self, space):
        pass

    def generate_parameters(self, *args, **kwargs):
        d = {"reproducer_info": "continue"}
        ATDDMessenger().write_tuner_info(d)
        params = self.params_info_list[self.counter]
        self.counter += 1
        if self.counter >= len(self.params_info_list):
            d = {"reproducer_info": "stop"}
            ATDDMessenger().write_tuner_info(d)
        return params

    def receive_trial_result(self, *args, **kwargs):
        pass
