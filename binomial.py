import math
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class Option(ABC):
    """
    An abstract Option base class that initialise common attributes of a stock option.
    They're inherited when its different child option pricing classes are instantiated.
    """
    S0: int
    K: int
    r: float = 0.05
    T: int = 1
    N: int = 2
    prob_u: Optional[float] = 0
    prob_d: Optional[float] = 0
    div: Optional[float] = 0
    sigma: float = 0
    is_put: bool = field(default=False)
    is_american: Optional[bool] = field(default=False)
    is_call: Optional[bool] = field(init=False)
    is_european: Optional[bool] = field(init=False)
    StockTrees: List[float] = field(init=False, repr=False, default_factory=list)
    """
    Initialise the stock option base dataclass, defaulted to European calls unless specified.
    
    Args:
        S0: Initial stock price
        K: Strike price
        r: Risk-free interest rate
        T: Time to maturity
        N: Number of time steps
    
    Optional Args:
        prob_u: Probability at up state
        prob_d: Probability at down state
        sigma: Volatility for CRR model
        div: Dividend yield
        is_put: True for a put option, False for a call option
        is_american: True for an American option, False for a European option
    """

    def __post_init__(self) -> None:
        self.is_call = not self.is_put
        self.is_european = not self.is_american
        
    @property
    def dt(self) -> float:
        """Single time step, in years"""
        return self.T/float(self.N)
    
    @property
    def df(self) -> float:
        """The discount factor """
        return math.exp(-(self.r - self.div) * self.dt)
    
    @abstractmethod
    def _define_u_and_d(self):
        """Setting up parameters depending on whether Binomial model is CRR or not."""
        raise NotImplementedError("NotImplementedError: Needs to be implemented")
    

class BinomialTree(Option):
    """
    This class allows for the pricing of an option by the Binomial tree model.
    It could either be a put/call and European/American. By default it prices a
    European call.
    """
    
    def __post_init__(self) -> None:
        super().__post_init__()
        self._define_u_and_d()
        self.qu = (math.exp((self.r - self.div) * self.dt) - self.d)/(self.u - self.d)
        self.qd = 1 - self.qu
    
    def _define_u_and_d(self) -> None:
        self.u = 1 + self.prob_u
        self.d = 1 - self.prob_d
    
    def _init_stock_price_tree(self) -> None:
        # Initialise a 2D tree at T=0
        self.StockTrees = [np.array([self.S0])]
        
        # Simulate the possible stock prices path
        for i in range(self.N):
            prev_branches = self.StockTrees[-1]
            st = np.concatenate((prev_branches*self.u, [prev_branches[-1]*self.d]))
            
            # Add nodes at each time step
            self.StockTrees.append(st)
    
    def _init_payoffs_tree(self) -> List[float]:
        if self.is_call:
            return np.maximum(0, self.StockTrees[self.N] - self.K)
        else:
            return np.maximum(0, self.K - self.StockTrees[self.N])
    
    def _check_early_exercise(self, payoffs, node) -> List[float]:
        if self.is_call:
            return np.maximum(payoffs, self.StockTrees[node] - self.K)
        else:
            return np.maximum(payoffs, self.K - self.StockTrees[node])
    
    def _traverse_tree(self, payoffs) -> List[float]:
        for i in reversed(range(self.N)):
            # The payoffs for European options from NOT exercising the option
            payoffs = (payoffs[:-1]*self.qu + payoffs[1:]*self.qd)*self.df
            
            # American options can be exercised early leading to different payoffs 
            if not self.is_european:
                payoffs = self._check_early_exercise(payoffs, i)
        return payoffs

    def _begin_tree_traversal(self) -> List[float]:
        payoffs = self._init_payoffs_tree()
        return self._traverse_tree(payoffs)
    
    def price(self) -> float:
        """
        Initiate the option pricing implementation
        """
        self._init_stock_price_tree()
        payoffs = self._begin_tree_traversal()
        
        # Present day option value converges back to the first node
        return payoffs[0]


class BinomialCRR(BinomialTree):
    """
    This class prices an option by the CRR binomial tree model.
    """
    def __post_init__(self) -> None:
        super().__post_init__()
        self._define_u_and_d()
        self.qu = (math.exp((self.r - self.div)*self.dt) - self.d)/(self.u - self.d)
        self.qd = 1 - self.qu        
    
    def _define_u_and_d(self) -> None:
        # Redefines the u and d for the CRR model
        self.u = math.exp(self.sigma * math.sqrt(self.dt))
        self.d = 1/self.u
    

if __name__ == "__main__":
    eu_option = BinomialTree(50, 52, r=0.05, T=2, N=2, prob_u=0.2, prob_d=0.2, is_put=True)
    print(eu_option.__repr__())
    print(f"Binomial Model European put option price is: {eu_option.price()}\n")
    
    am_option = BinomialTree(50, 52, r=0.05, T=2, N=2, prob_u=0.2, prob_d=0.2, is_put=True, is_american=True)
    print(am_option.__repr__())
    print(f"Binomial Model American put option price is: {am_option.price()}\n")
    
    eu_option_crr = BinomialCRR(50, 52, r=0.05, T=2, N=2, sigma=0.3, is_put=True)
    print(eu_option_crr.__repr__())
    print(f"CRR Binomial Model European put option price is: {eu_option_crr.price()}\n")
    
    am_option_crr = BinomialCRR(50, 52, r=0.05, T=2, N=2, sigma=0.3, is_put=True, is_american=True)
    print(eu_option_crr.__repr__())
    print(f"CRR Binomial Model American put option price is: {am_option_crr.price()}\n")