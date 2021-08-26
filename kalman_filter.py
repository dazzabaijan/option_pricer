import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader.data as web
from dataclasses import dataclass, field
import inspect

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
    
    def _initialise_parameters(self):
        arguments = get_params(self)
        print(arguments)
    
    def filter(self, X):
        Z = X.reshape(len(X), 1)
        arguments = self._initialise_parameters()

        n_timesteps = X.shape[0]
        n_dim_state = len(self.initial_state_mean)
        n_dim_obs = X.shape[1]
        predicted_state_means = np.zeros((n_timesteps, n_dim_state))
        predicted_state_covariances = np.zeros((n_timesteps, n_dim_state, n_dim_state))
        kalman_gains = np.zeros((n_timesteps, n_dim_state, n_dim_obs))
        filtered_state_means = np.zeros((n_timesteps, n_dim_state))
        filtered_state_covariances = np.zeros((n_timesteps, n_dim_state, n_dim_state))
        
        for t in range(n_timesteps):
            if t == 0:
                predicted_state_means[t] = self.initial_state_mean
                predicted_state_covariances[t] = self.initial_state_covariance
            else:
                predicted_state_means[t], predicted_state_covariances[t] = \
                    _filter_predict()
    
    @staticmethod
    def _filter_predict(
        transition_matrix,
        transition_covariance,
        transition_offset,
        current_state_mean,
        current_state_covariance):
        
        predicted_state_mean = transition_matrix@current_state_mean + transition_offset
        
        predicted_state_covariance = transition_matrix@current_state_covariance@transition_matrix.T \
            + transition_covariance
        
        return (predicted_state_mean, predicted_state_covariance)


def get_params(obj):
    args = inspect.getfullargspec(obj.__init__)[0]
    args.pop(0)
    
    argdict = dict([(arg, obj.__getattribute__(arg)) for arg in args])
    return argdict


if __name__ == "__main__":
    etfs = ['TLT', 'IEI']
    start_date = "2010-8-01"
    end_date = "2016-08-01"
    
    prices = web.DataReader(
        etfs, 'yahoo', start_date, end_date
    )['Adj Close']
    
    delta = 1e-5
    trans_cov = delta / (1 - delta) * np.eye(2)
    obs_mat = np.vstack(
        [prices[etfs[0]], np.ones(prices[etfs[0]].shape)]
    ).T[:, np.newaxis]
    kf = KalmanFilter(
        n_dim_obs=1, 
        n_dim_state=2,
        initial_state_mean=np.zeros(2),
        initial_state_covariance=np.ones((2, 2)),
        transition_matrices=np.eye(2),
        observation_matrices=obs_mat,
        observation_covariance=1.0,
        transition_covariance=trans_cov
    )
    print(prices[etfs[1]].values.reshape(len(prices[etfs[1]].values), 1).shape)
    # argdict = kf.filter(prices[etfs[1]].values)