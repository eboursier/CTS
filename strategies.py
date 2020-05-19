from __future__ import print_function
import numpy as np
import os
import networkx as nx
import itertools

class BanditAlgo():
    """
    Superclass for bandit algo/strategies.
    An algo must have three functions: - self.action() returns the arms to pull for the next timestep
                                       - self.update(plays, feedback) (where feedback is exactly X[plays]) updates the strategy when the algo pulled the arms given by plays
                                        and received the semi-bandit feedback given by feedbakc, feedback[i] is exactly the statistic of the arm plays[i].
                                       - self.reset() reset the algo to 0 (forget the past observations)
    """
    def __init__(self, name):
        self.name = name


######## UCB

class CUCB(BanditAlgo):
    """
    CUCB algo.
    """
    def __init__(self, n, oracle, sigma=1, delta=lambda t:max(1, np.log(t)+3*np.log(np.log(t)))):
        super().__init__("CUCB")
        self.means = np.zeros(n) # empirical mean
        self.n = n
        self.N = np.zeros(n) # number of pulls for each arm
        self.oracle = oracle
        self.t = 0
        self.delta = delta # used for the confidence bound
        self.sigma = sigma # subgaussian parameter (1/2 for variables in [0,1], 1 for gaussian with variance 1)

    def action(self):
        self.t += 1
        return self.oracle.action(self.means + self.sigma*np.sqrt(2*self.delta(self.t)/self.N))

    def update(self, plays, feedback):
        self.means[plays] = (self.means[plays]*self.N[plays]+feedback)/(self.N[plays]+1) # update empirical means
        self.N[plays] += 1 # update counters

    def reset(self):
        self.t = 0
        self.means = np.zeros(self.n)
        self.N = np.zeros(self.n)


class CUCBKLPath(BanditAlgo):
    """
    CUCB-KL algo (better for binary variables) for shortest path problem (deal with means in [-1, 0])
    """
    def __init__(self, n, oracle, eps=1e-5, precision=1e-6, max_iter=1e3, delta=lambda t:max(1, np.log(t)+3*np.log(np.log(t)))):
        super().__init__("CUCB-KL")
        self.means = np.zeros(n) # empirical mean
        self.n = n
        self.N = np.zeros(n) # number of pulls for each arm
        self.oracle = oracle
        self.t = 0
        self.delta = delta # used for the confidence bound
        self.eps = eps
        self.precision = precision
        self.max_iter = int(max_iter)

    def action(self):
        self.t += 1
        return self.oracle.action([self.inv_kl[i] for i in range(self.n)])

    def kl(self, x, y):
        x = np.min(np.max(x, self.eps), 1 - self.eps)
        y = np.min(np.max(y, self.eps), 1 - self.eps)
        return x * np.log(x / y) + (1 - x) * np.log((1 - x) / (1 - y))

    def inv_kl(self, i):
        """
        return the KL upper bound for arm i
        """
        val = self.means[i] # approximation by below
        u = 0 # approximation by above (negative means)
        _iter = 0
        while iter < self.max_iter and u-val > self.precision: # look for the largest x such that N_i kl(mean_i, x) < beta*log(t)
            _iter += 1
            m = (val + u)/2
            if self.kl(1+self.means[i], 1+m) > self.delta(self.t)/self.N[i]:
                u = m
            else:
                val = m
        return (val + u)/2

    def update(self, plays, feedback):
        self.means[plays] = (self.means[plays]*self.N[plays]+feedback)/(self.N[plays]+1) # update empirical means
        self.N[plays] += 1 # update counters

    def reset(self):
        self.t = 0
        self.means = np.zeros(self.n)
        self.N = np.zeros(self.n)


######## Thompson Sampling

class CTSBetaPath(BanditAlgo):
    """
    Implement CTS-Beta algorithm for shortest path algorithm (has to deal with negative means)
    """
    def __init__(self, oracle, n, init_count=1, a=1, b=1):
        super().__init__("Optimistic CTS-Beta")
        self.a = a*np.ones(n)
        self.b = b*np.ones(n)
        self.n = n
        self.init_count = init_count # in itialization as long as all arms have not been pulled init_count times
        self.init = True # True if we are still initialization
        self.means = np.zeros(n) # (means, N) can be deduced from (a, b)
        self.N = np.zeros(n) # but we add them to track the algo history/statistics
        self.oracle = oracle

    def update(self, plays, feedback):
        N = self.N[plays]
        self.means[plays] = (self.means[plays]*N + feedback)/(N+1) # update empirical mean
        self.a[feedback==-1] += 1 # update posterior (negative rewards)
        self.b[feedback==0] += 1 # update posterior
        self.N[plays] += 1 # update the counters

        if self.init:
            self.init = np.sum(self.N<self.init_count) >= 1 # end init. if all arms have been pulled at least init_count times

    def action(self):
        if self.init:
            theta = (self.N<self.init_count)-1 # ensure that the oracle will pull arms that have not been pulled enough yet.
        else:
            theta = -np.random.beta(self.a, self.b) # beta sample
            theta = np.max(theta, self.means) # optimistic sampling
        self.t += 1
        return self.oracle.action(theta)


class CTSGaussian(BanditAlgo):
    """
    Implement CTS-Gaussian algorithm with its specific update rule.
    CTS-Gaussian is here implemented to handle non independent prior for generality reasons.
    """
    def __init__(self, means, cov, oracle, init_count=1):
        super().__init__("CTS-Gaussian") 
        self.init_means = means
        self.subg_matrix = cov # subgaussianity matrix
        self.means = means
        self.cov = np.zeros_like(cov)
        self.oracle = oracle # the reward function and action sets are 'specified' in the oracle
        self.init_count = init_count # in itialization as long as all arms have not been pulled init_count times
        self.init = True # True if we are still initialization
        self.t = 0 # timestep of algo
        self.N = np.zeros((len(means), len(means))) # n_ij is the number of times i,j got played together

    def update(self, plays, feedback):
        """
        Update the posterior distribution after playing plays and receiving feedback = X[plays]
        """

        N = self.N[np.meshgrid(plays, plays)]
        self.means[plays] = (self.means[plays]*np.diag(N) + feedback)/(np.diag(N)+1) # update empirical mean
        # sigma_ij = gamma_ij n_ij / (n_i n_j) here in the non independent case (does not change the diagonal terms in independent case)
        self.cov[np.meshgrid(plays, plays)] = self.subg_matrix[np.meshgrid(plays, plays)] * (np.diag(1/(np.diag(N)+1)) @ (N+1) @ np.diag(1/(np.diag(N)+1))) # update covariance matrix
        self.N[np.meshgrid(plays, plays)] += 1 # update the counters

        if self.init:
            self.init = np.sum(np.diag(self.N)<self.init_count) >= 1 # end init. if all arms have been pulled at least init_count times
    
    def action(self):
        """
        Return Oracle(theta_t).
        """
        if self.init:
            # watch out !!! in very specific cases, the following line might not be sufficient to ensure the initialization is complete
            theta = (np.diag(self.N)<self.init_count)+0 # ensure that the oracle will pull arms that have not been pulled enough yet.
        else:
            theta = np.random.multivariate_normal(self.means, self.cov)
        self.t += 1
        return self.oracle.action(theta)

    def reset(self):
        self.N = np.zeros((len(self.means), len(self.means)))
        self.means = np.copy(self.init_means)
        self.cov = np.zeros_like(self.cov)
        self.init = True
        self.t = 0


class clipCTSGaussian(BanditAlgo):
    """
    Implement CTS-Gaussian algorithm with its specific update rule.
    CTS-Gaussian implemented for INDEPENDENT prior only (for speed up of the algo). See (commented) version above for general algo.
    """
    def __init__(self, means, cov, oracle, init_count=1, sigma=1, delta=lambda t:max(1, np.log(t)+3*np.log(np.log(t)))):
        if not(np.allclose(np.diag(np.diag(cov)),cov)):
            raise ValueError('The Subgaussian Matrix has to be diagonal') # in practice, we might prefer to "diagonalize" and regularize cov at this step. Can be easily modified
        super().__init__("Clip CTS-Gaussian")
        self.init_means = means
        self.means = means
        self.oracle = oracle # the reward function and action sets are 'specified' in the oracle
        self.init_count = init_count # in itialization as long as all arms have not been pulled init_count times
        self.init = True # True if we are still initialization
        self.t = 0 # timestep of algo
        self.N = np.zeros(len(means)) # n_ij is the number of times i,j got played together
        self.subg_var = np.diag(cov)
        self.cov = np.zeros(len(means)) # diagonal of the covariance of Theta
        self.delta = delta
        self.sigma = sigma

    def update(self, plays, feedback):
        """
        Update the posterior distribution after playing plays and receiving feedback = X[plays]
        """
        N = self.N[plays]
        self.means[plays] = (self.means[plays]*N + feedback)/(N+1) # update empirical mean
        self.cov[plays] = self.subg_var[plays]/(N+1) # update variances
        self.N[plays] += 1 # update the counters

        if self.init:
            self.init = np.sum(self.N<self.init_count) >= 1 # end init. if all arms have been pulled at least init_count times

    def action(self):
        """
        Return Oracle(theta_t).
        """
        self.t += 1
        if self.init:
            # watch out !!! in very specific cases, the following line might not be sufficient to ensure the initialization is complete
            theta = (self.N<self.init_count)+0 # ensure that the oracle will pull arms that have not been pulled enough yet.
        else:
            theta = np.random.multivariate_normal(self.means, np.diag(self.cov)) # independent prior here
            theta = np.clip(theta, self.means, self.means + self.sigma*np.sqrt(2*self.delta(self.t)/self.N))
        return self.oracle.action(theta)

    def reset(self):
        self.N = np.zeros(len(self.means))
        self.means = np.copy(self.init_means)
        self.cov = np.zeros_like(self.cov)
        self.init = True
        self.t = 0

######## ESCB

class ESCBCompleteMatching(BanditAlgo):
    """
    ESCB algo for matching problem when the bipartite graph is a complete graph. 
    Warning ! It has a combinatorial complexity and can not manage large number of arms.
    """
    def __init__(self, g, cov):
        super().__init__("ESCB")
        self.n = len(g.edges)
        self.left, self.right = nx.bipartite.sets(self.graph)
        self.m = min(len(left), len(right))
        self.means = np.zeros(self.n) # empirical mean
        self.N = np.zeros((self.n,self.n)) # n_ij is the number of times that i and j got simultaneously pulled
        self.t = 0
        self.delta = lambda t:np.log(t) + (self.m + 2)*np.log(np.log(t)) + 0.5*self.m*np.log(1+np.e) # used for the confidence bound
        self.subg_matrix = cov # subgaussian parameter (1/2 for variables in [0,1], 1 for gaussian with variance 1)
        self.cov = np.inf*np.ones_like(cov)
        self.graph = g.copy()
        self.edge_dict = {}
        for i, (u,v) in enumerate(self.graph.edges):
            self.edge_dict[np.min((u,v)),np.max((u,v))] = i

    def matching_to_indices(self, match):
        plays = [self.edge_dict[np.min((u, v)), np.max((u, v))] for u,v in match]
        return np.array(plays)

    def action(self):
        """
        Compute the action brute force. This might take a lot of time, especially when the number of arms is large.
        """
        self.t += 1
        max_val = -np.inf
        for i, right in enumerate(itertools.permutations(self.right)):
            match = [(self.left[i], right[i]) for i in len(self.m)] # check for all possible matchings
            plays = self.matching_to_indices(match)
            val = np.sum(self.means[plays]) + np.sqrt(2*self.delta(self.t)*np.sum(self.cov[np.meshgrid(plays, plays)])) # upper confidence bound for this matching
            if val > max_val
                max_val = val
                max_match = plays
        return max_match

    def update(self, plays, feedback):
        N = self.N[np.meshgrid(plays, plays)]
        self.means[plays] = (self.means[plays]*np.diag(N) + feedback)/(np.diag(N)+1) # update empirical mean
        self.cov[np.meshgrid(plays, plays)] = self.subg_matrix[np.meshgrid(plays, plays)] * (np.diag(1/(np.diag(N)+1)) @ (N+1) @ np.diag(1/(np.diag(N)+1))) # update covariance matrix
        self.N[np.meshgrid(plays, plays)] += 1 # update the counters

    def reset(self):
        self.t = 0
        self.means = np.zeros(self.n)
        self.N = np.zeros(self.n)
        self.cov = np.inf*np.ones_like(self.subg_matrix)

if __name__=='__main__':
    pass