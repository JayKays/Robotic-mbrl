
import numpy as np
from matplotlib import pyplot as plt

import pandas as pd
import mbrl
import pathlib

import mbrl.util.common

from scipy.signal import medfilt


def median_filter(data, window_size = 5):
    '''
    Performs a running median filtering of the input data array along each time axis
    '''

    filtered_data = medfilt(data.copy(), kernel_size=(window_size,))

    return filtered_data

def load_uncertainty_log(dir):
    path = dir + "/uncertainty_log.npz"

    return np.load(path)

def load_results(dir):
    path = dir + "/results.csv"

    return pd.read_csv(path)

def load_buffer(dir):
    return np.load(dir + "/replay_buffer.npz")

def load_model(model_dir):
    model_path = pathlib.Path(model_dir)
    cfg = mbrl.util.common.load_hydra_cfg(model_path)
    handler = mbrl.util.create_handler(cfg)

    env, _, _ = handler.make_env(cfg)

    model = mbrl.util.common.create_one_dim_tr_model(
        cfg,
        env.observation_space.shape,
        env.action_space.shape,
        model_dir=model_path,
        )
    return model