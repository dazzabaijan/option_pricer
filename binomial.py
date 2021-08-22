import math
import numpy as np
from decimal import Decimal
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class Option(ABC):
    """
    An Option base class that stores common attributes of a stock option
    """
    S0: int
    K: int
    r: float = 0.05
    T: int = 1
    N: int = 2
    StockTrees: List[float] = field(init=False, repr=False, default_factory=list)
    pu: Optional[float] = 0
    pd: Optional[float] = 0
    div: Optional[float] = 0
    sigma: float = 0
    is_put: bool = field(default=False)
    is_american: Optional[bool] = field(default=False)
    is_call: Optional[bool] = field(init=False)
    is_european: Optional[bool] = field(init=False)
    
    def __post_init__(self):
        self.is_call = not self.is_put
        self.is_european = not self.is_american
        
    @property
    def dt(self):
        """Single time step, in years"""
        return self.T/float(self.N)
    
    @property
    def df(self):
        """The discount factor """
        return math.exp(-(self.r - self.div) * self.dt)
    
    @abstractmethod
    def _define_u_and_d(self):
        """Setting up parameters depending on whether Binomial model is CRR or not."""
        raise NotImplementedError("NotImplementedError: Needs to be implemented")


class BinomialEuropean(Option):
    """
    Price a European option by the binomial tree model
    """
    def setup_parameters(self):
        # Required calculations for the model
        self.M = self.N + 1 # Number of terminal nodes of tree
        self.u = 1 + self.pu # Expected value in the up state
        self.d = 1 - self.pd # Expected value in the down state
        self.qu = (math.exp((self.r - self.div) * self.dt) - self.d)/(self.u - self.d)
        self.qd = 1 - self.qu
    
    def init_stock_price_tree(self):
        # Initialise terminal price nodes to zeros
        self.StockTrees = np.zeros(self.M)
        
        # Calculate expected stock prices for each node
        for i in range(self.M):
            self.StockTrees[i] = self.S0 * (self.u**(self.N - i)) * (self.d**i)
            print(f"self.StockTrees[{i}] = {self.StockTrees[i]}")
    
    def init_payoffs_tree(self):
        """
        Returns the payoffs when the option expires at terminal nodes
        """
        print(f"{self.is_call=}")
        if self.is_call:
            print(f"{np.maximum(0, self.StockTrees - self.K) = }")
            return np.maximum(0, self.StockTrees - self.K)
        else:
            print(f"{np.maximum(0, self.K - self.StockTrees) = }")
            return np.maximum(0, self.K - self.StockTrees)
    
    def traverse_tree(self, payoffs):
        """
        Starting from the time the option expires, traverse backwards
        and calculate discounted payoffs at each node
        """
        for _ in range(self.N):
            payoffs = (payoffs[:-1] * self.qu + payoffs[1:] * self.qd) * self.df
        print(f"{payoffs=}")
        return payoffs
    
    def begin_tree_traversal(self):
        payoffs = self.init_payoffs_tree()
        return self.traverse_tree(payoffs)
    
    def price(self):
        """
        Entry point of the pricing implementation
        """
        self.setup_parameters()
        self.init_stock_price_tree()
        payoffs = self.begin_tree_traversal()
        
        # Option value converges to first node
        return payoffs[0]
    

class BinomialTree(Option):
    """
    Price a American option by the binomial tree model
    """
    u: float = field(init=False)
    d: float = field(init=False)
    qu: float = field(init=False)
    qd: float = field(init=False)
    
    def __post_init__(self):
        super().__post_init__()
        self._define_u_and_d()
        self.qu = (math.exp((self.r - self.div) * self.dt) - self.d)/(self.u - self.d)
        self.qd = 1 - self.qu
    
    def _define_u_and_d(self):
        self.u = 1 + self.pu
        self.d = 1 - self.pd
    
    def init_stock_price_tree(self):
        # Initialise a 2D tree at T=0
        self.StockTrees = [np.array([self.S0])]
        
        # Simulate the possible stock prices path
        for i in range(self.N):
            prev_branches = self.StockTrees[-1]
            st = np.concatenate((prev_branches*self.u, [prev_branches[-1]*self.d]))
            self.StockTrees.append(st) # Add nodes at each time step
    
    def init_payoffs_tree(self):
        if self.is_call:
            return np.maximum(0, self.StockTrees[self.N] - self.K)
        else:
            return np.maximum(0, self.K - self.StockTrees[self.N])
    
    def check_early_exercise(self, payoffs, node):
        if self.is_call:
            return np.maximum(payoffs, self.StockTrees[node] - self.K)
        else:
            return np.maximum(payoffs, self.K - self.StockTrees[node])
    
    def traverse_tree(self, payoffs):
        print(f"{self.is_european = } ")
        for i in reversed(range(self.N)):
            # The payoffs from NOT exercising the option
            payoffs = (payoffs[:-1]*self.qu + payoffs[1:]*self.qd)*self.df
            
            # Payoffs from exercising, for American options
            if not self.is_european:
                payoffs = self.check_early_exercise(payoffs, i)
        return payoffs

    def begin_tree_traversal(self):
        payoffs = self.init_payoffs_tree()
        return self.traverse_tree(payoffs)
    
    def price(self):
        """
        Entry point of the pricing implementation
        """
        # self.setup_parameters()
        self.init_stock_price_tree()
        payoffs = self.begin_tree_traversal()
        
        # Option value converges to first node
        return payoffs[0]


class BinomialCRROption(BinomialTree):
    
    def __post_init__(self):
        super().__post_init__()
        self._define_u_and_d()
        self.qu = (math.exp((self.r - self.div)*self.dt) - self.d)/(self.u - self.d)
        self.qd = 1 - self.qu        
    
    def _define_u_and_d(self):
        self.u = math.exp(self.sigma * math.sqrt(self.dt))
        self.d = 1/self.u
    

if __name__ == "__main__":
    # eu_option = BinomialEuropean(50, 52, r=0.05, T=2, N=2, pu=0.2, pd=0.2, is_put=True)
    # 
    # print(f"European put option price is: {eu_option.price()}")
    eu_option = BinomialTree(50, 52, r=0.05, T=2, N=2, pu=0.2, pd=0.2, is_put=True)
    print(eu_option.__repr__())
    print(f"Binomial Mode lEuropean put option price is: {eu_option.price()}")
    
    am_option = BinomialTree(50, 52, r=0.05, T=2, N=2, pu=0.2, pd=0.2, is_put=True, is_american=True)
    print(am_option.__repr__())
    print(f"Binomial Model American put option price is: {am_option.price()}")
    
    eu_option_crr = BinomialCRROption(50, 52, r=0.05, T=2, N=2, sigma=0.3, is_put=True)
    print(eu_option_crr.__repr__())
    print(f"CRR Binomial Model European put option price is: {eu_option_crr.price()}")
    
    am_option_crr = BinomialCRROption(50, 52, r=0.05, T=2, N=2, sigma=0.3, is_put=True, is_american=True)
    print(eu_option_crr.__repr__())
    print(f"CRR Binomial Model American put option price is: {am_option_crr.price()}")