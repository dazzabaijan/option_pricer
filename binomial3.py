import math
import numpy as np
from decimal import Decimal
from abc import ABC, abstractmethod


class Option(object):
    """
    Stores common attributes of a stock option
    """
    def __init__(
        self,
        S0: int,
        K: int,
        r=0.05,
        T=1,
        N=2,
        pu=0,
        pd=0,
        div=0,
        sigma=0,
        is_put=False,
        is_american=False
    ) -> int:
        """
        Initialise the stock option base class.
        Defaults to European calls unless specified.
        
        Arg:
            S0: Initial stock price
            K: Strike price
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
        self.K = K
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
    
    @abstractmethod
    def setup_parameters(self):
        raise NotImplementedError("Needs to be implemented")
    

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
            return np.maximum(0, self.StockTrees - self.K)
        else:
            return np.maximum(0, self.K - self.StockTrees)
    
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
    

class BinomialTree(Option):
    """
    Price a American option by the binomial tree model
    """
    def setup_parameters(self):
        # Required calculations for the model
        self.u = 1 + self.pu # Expected value in the up state
        self.d = 1 - self.pd # Expected value in the down state
        self.qu = (math.exp((self.r - self.div) * self.dt) - self.d)/(self.u - self.d)
        self.qd = 1 - self.qu

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
        self.setup_parameters()
        self.init_stock_price_tree()
        payoffs = self.begin_tree_traversal()
        
        # Option value converges to first node
        return payoffs[0]

class BinomialCRROption(BinomialTree):
    def setup_parameters(self):
        self.u = math.exp(self.sigma * math.sqrt(self.dt))
        self.d = 1/self.u
        self.qu = (math.exp((self.r - self.div)*self.dt) - self.d)/(self.u - self.d)
        self.qd = 1 - self.qu
    

if __name__ == "__main__":
    eu_option = BinomialEuropean(50, 52, r=0.05, T=2, N=2, pu=0.2, pd=0.2, is_put=True)
    print(f"European put option price is: {eu_option.price()}")
    
    am_option = BinomialTree(50, 52, r=0.05, T=2, N=2, pu=0.2, pd=0.2, is_put=True, is_american=True)
    print(f"American put option price is: {am_option.price()}")
    
    eu_option2 = BinomialCRROption(50, 52, r=0.05, T=2, N=2, sigma=0.3, is_put=True)
    print(f"European put: {eu_option2.price()}")
    
    am_option2 = BinomialCRROption(50, 52, r=0.05, T=2, N=2, sigma=0.3, is_put=True, is_american=True)
    print(f"American put option price is: {am_option2.price()}")