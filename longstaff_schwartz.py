import numpy as np
from scipy.stats import norm

class GeometricBrownianMotion:
    
    def __init__(
        self,
        s_0: float,
        mu: float,
        sigma: float,
        n_simulations: int,
        T: float,
        N: int,
        seed: int
    ) -> np.ndarray:
    
        self.s_0 = s_0
        self.mu = mu
        self.sigma = sigma
        self.n_simulations = n_simulations
        self.T = T
        self.N = N
        self.seed = seed
        np.random.seed(self.seed)
    
    @property
    def dt(self) -> float:
        """Single time step, in years"""
        return self.T/float(self.N)
    
    def simulate(self) -> np.ndarray:
        
        dW = np.random.normal(scale=np.sqrt(self.dt), size=(self.n_simulations, self.N+1))
        
        # Cumulative sum of the Brownian increments dW gives the Brownian path
        W = np.cumsum((self.mu - 0.5*self.sigma**2)*self.dt + self.sigma*dW, axis=1)
        S_t = self.s_0*np.exp(W)
        S_t[:, 0] = self.s_0
        
        return S_t

if __name__ == "__main__":
    gbm = GeometricBrownianMotion(s_0=36, mu=0.06, sigma=0.2, n_simulations=100, T=1, N=50, seed=42)
    print(gbm.simulate())
    print(len(gbm.simulate()))