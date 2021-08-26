import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader.data as web
from dataclasses import dataclass, field

@dataclass
class KalmanFilter(object):
    """First attempt at implementing a basic Kalman filter"""
    n_dim_obs: int
    n_dim_state: int
    initial_state_mean: np.ndarray
    initial_state_covariance: np.ndarray
    transition_matrices: np.ndarray
    observation_matrices: np.ndarray
    observation_covariance: float
    transition_covariance: float
    