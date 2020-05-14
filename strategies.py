from __future__ import print_function
import numpy as np
import os

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
    def __init__(self, K, oracle, alpha=6):
        super().__init__("CUCB")
        self.means = np.zeros(K) # empirical mean
        self.K = K
        self.N = np.zeros(K) # number of pulls for each arm
        self.oracle = oracle
        self.t = 0
        self.alpha = alpha # hyper parameter for the confidence bound (we set 4 x the original paper as we here use 1-subgaussian instead of 1/4-subgaussian)

    def action(self):
        self.t += 1
        return self.oracle.action(self.means + np.sqrt(self.alpha*np.log(self.t)/self.N))

    def update(self, plays, feedback):
        self.means[plays] = (self.means[plays]*self.N[plays]+feedback)/(self.N[plays]+1) # update empirical means
        self.N[plays] += 1 # update counters

    def reset(self):
        self.t = 0
        self.means = np.zeros(self.K)
        self.N = np.zeros(self.K)


######## Thompson Sampling

class GaussianThompsonSampling(BanditAlgo):
    """
    Superclass for combinatorial thompson sampling algos.
    Specific classes are added below to allow different update rules of the posterior.
    """
    def __init__(self, name, means, cov, oracle, init_count):
        """
        means and cov are the means and covariance of the gaussian prior
        """
        super().__init__(name)
        self.init_means = means
        self.subg_matrix = cov # subgaussianity matrix
        self.means = means
        self.cov = np.zeros_like(cov)
        self.oracle = oracle # the reward function and action sets are 'specified' in the oracle
        self.init_count = init_count # in itialization as long as all arms have not been pulled init_count times
        self.init = True # True if we are still initialization
        self.t = 0 # timestep of algo

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

class CTSGaussian(GaussianThompsonSampling):
    """
    Implement CTS-Gaussian algorithm with its specific update rule.
    CTS-Gaussian is here implemented to handle non independent prior for generality reasons.
    """
    def __init__(self, means, cov, oracle, init_count=1):
        super().__init__("CTS-Gaussian", means, cov, oracle, init_count) 
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

    def reset(self):
        self.N = np.zeros((len(self.means), len(self.means)))
        self.means = np.copy(self.init_means)
        self.cov = np.zeros_like(self.cov)
        self.init = True
        self.t = 0


class CTSGaussianIndep(GaussianThompsonSampling):
    """
    Implement CTS-Gaussian algorithm with its specific update rule.
    CTS-Gaussian implemented for INDEPENDENT prior only (for speed up of the algo). See version above for general algo.
    """
    def __init__(self, means, cov, oracle, init_count=1):
        if not(np.allclose(np.diag(np.diag(cov)),cov)):
            raise ValueError('The Subgaussian Matrix has to be diagonal') # in practice, we might prefer to "diagonalize" and regularize cov at this step. Can be easily modified
        super().__init__("CTS-GaussianIndep", means, cov, oracle, init_count) 
        self.N = np.zeros(len(means)) # n_ij is the number of times i,j got played together
        self.subg_var = np.diag(self.subg_matrix)
        self.cov = np.zeros(len(means)) # diagonal

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
        if self.init:
            # watch out !!! in very specific cases, the following line might not be sufficient to ensure the initialization is complete
            theta = (self.N<self.init_count)+0 # ensure that the oracle will pull arms that have not been pulled enough yet.
        else:
            theta = np.random.multivariate_normal(self.means, np.diag(self.cov)) # independent prior here
        self.t += 1
        return self.oracle.action(theta)

    def reset(self):
        self.N = np.zeros(len(self.means))
        self.means = np.copy(self.init_means)
        self.cov = np.zeros_like(self.cov)
        self.init = True
        self.t = 0

if __name__ == '__main__':
    pass