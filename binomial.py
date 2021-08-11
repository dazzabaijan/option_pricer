import math
import numpy as np
from decimal import Decimal

class Option(object):
    """
    Stores common attributes of a stock option
    """
    def __init__(self, S0, strike, r=0.05, T=1, N=2, pu=0, pd=0, div=0, sigma=0, is_put=False, is_american=False):
        """
        Initialise the stock option base class.
        Defaults to European calls unless specified.
        
        Arg:
            S0: Initial stock price
            strike: Strike price
            r: Risk-free interest rate
            T: Time to maturity
            N: Number of time steps
            pu: Probability at up state
            pd: Probability at down state
            div: Dividend yield
            is_put: True for a put option, False for a call option
            is_american: True for an American option, False for a European option
        """
        self.S0 = S0
        self.strike = strike
        self.r = r
        self.T = T
        self.N = max(1, N)
        self.STs = []  # Declare the stock prices tree
        
        """Optional parameters used by derived classes"""
        self.pu, self.pd = pu, pd
        self.div = div
        self.sigma = sigma
        self.is_call = not is_put
        self.is_european = not is_american
        
    @property
    def dt(self):
        """Single time step, in years"""
        return self.T/float(self.N)
    
    @property
    def df(self):
        """The discount factor """
        return math.exp(-(self.r - self.div) * self.dt)
    

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
    
    def init_payoffs_tree(self):
        """
        Returns the payoffs when the option expires at terminal nodes
        """
        if self.is_call:
            return np.maximum(0, self.StockTrees - self.strike)
        else:
            return np.maximum(0, self.strike - self.StockTrees)
    
    def traverse_tree(self, payoffs):
        """
        Starting from the time the option expires, traverse backwards
        and calculate discounted payoffs at each node
        """
        for i in range(self.N):
            payoffs = (payoffs[:-1] * self.qu + payoffs[1:] * self.qd) * self.df
        
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