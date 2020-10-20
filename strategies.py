from __future__ import print_function
import numpy as np
import os
import networkx as nx
import itertools
from utils import *


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


# UCB

class CUCB(BanditAlgo):
    """
    CUCB algo.
    """

    def __init__(self, n, oracle, sigma=1,
                 delta=lambda t: max(1, np.log(t)+3*np.log(np.log(t)))):
        super().__init__("CUCB")
        self.means = np.zeros(n)  # empirical mean
        self.n = n
        self.N = np.zeros(n)  # number of pulls for each arm
        self.oracle = oracle
        self.t = 0
        self.delta = delta  # used for the confidence bound
        self.sigma = sigma  # subgaussian parameter

    def action(self):
        self.t += 1
        conf_bound = self.sigma*np.sqrt(2*self.delta(self.t)/self.N)
        return self.oracle.action(self.means + conf_bound)

    def update(self, plays, feedback):
        # update empirical means
        self.means[plays] = (
            self.means[plays]*self.N[plays]+feedback)/(self.N[plays]+1)
        self.N[plays] += 1  # update counters

    def reset(self):
        self.t = 0
        self.means = np.zeros(self.n)
        self.N = np.zeros(self.n)


class CUCBKLPath(BanditAlgo):
    """
    CUCB-KL algo (better for binary variables) for shortest path problem (deal with means in [-1, 0])
    """

    def __init__(self, n, oracle, eps=1e-5, precision=1e-6, max_iter=1e3,
                 delta=lambda t: max(1, np.log(t)+3*np.log(np.log(t)))):
        super().__init__("CUCB-KL")
        self.means = np.zeros(n)  # empirical mean
        self.n = n
        self.N = np.zeros(n)  # number of pulls for each arm
        self.oracle = oracle
        self.t = 0
        self.delta = delta  # used for the confidence bound
        self.eps = eps
        self.precision = precision
        self.max_iter = int(max_iter)

    def action(self):
        self.t += 1
        return self.oracle.action([self.inv_kl(i) for i in range(self.n)])

    def kl(self, x, y):
        x = np.clip(x, self.eps, 1 - self.eps)
        y = np.clip(y, self.eps, 1 - self.eps)
        return x * np.log(x / y) + (1 - x) * np.log((1 - x) / (1 - y))

    def inv_kl(self, i):
        """
        return the KL upper bound for arm i
        """
        val = self.means[i]  # approximation by below
        u = 0  # approximation by above (negative means)
        _iter = 0
        # look for the largest x such that N_i kl(mean_i, x) < beta*log(t)
        while _iter < self.max_iter and u-val > self.precision:
            _iter += 1
            m = (val + u)/2
            if self.kl(1+self.means[i], 1+m) > self.delta(self.t)/self.N[i]:
                u = m
            else:
                val = m
        return (val + u)/2

    def update(self, plays, feedback):
        # update empirical means
        self.means[plays] = (self.means[plays]*self.N[plays]+feedback) / \
            (self.N[plays]+1)
        self.N[plays] += 1  # update counters

    def reset(self):
        self.t = 0
        self.means = np.zeros(self.n)
        self.N = np.zeros(self.n)


# Thompson Sampling

class CTSBetaPath(BanditAlgo):
    """
    Implement CTS-Beta algorithm for shortest path algorithm (has to deal with negative means)
    """

    def __init__(self, n, oracle, init_count=1, optimistic=True):
        if optimistic:
            super().__init__("Optimistic CTS-Beta")
        else:
            super().__init__("CTS-Beta")
        self.optimistic = optimistic
        self.a = np.ones(n)
        self.b = np.ones(n)
        self.n = n
        self.means = np.zeros(n)  # (means, N) can be deduced from (a, b)
        # but we add them to track the algo history/statistics
        self.N = np.zeros(n)
        self.oracle = oracle

    def update(self, plays, feedback):
        N = self.N[plays]
        self.means[plays] = (self.means[plays]*N + feedback) / \
            (N+1)  # update empirical mean
        # update posterior (negative rewards)
        self.a[plays[feedback == -1]] += 1
        self.b[plays[feedback == 0]] += 1  # update posterior
        self.N[plays] += 1  # update the counters

    def action(self):

        theta = -np.random.beta(self.a, self.b)  # beta sample
        if self.optimistic:
            theta = np.max((theta, self.means), axis=0)  # optimistic sampling
        self.t += 1
        return self.oracle.action(theta)

    def reset(self):
        self.N = np.zeros(self.n)
        self.means = np.zeros(self.n)
        self.a = np.ones(self.n)
        self.b = np.ones(self.n)
        self.init = True
        self.t = 0


class CorrelatedPrior(BanditAlgo):
    """
    Implement CTS-Gaussian algorithm with correlated prior.
    posterior: "ellipse" or "app-ellipse". Determine the posterior covariance to use.
                - "ellipse" gives cov_ij = Gamma_ij n_ij/(n_i*n_j) (default choice)
                - "app-ellipse" gives cov_ij = Gamma_ij/(sqrt(n_i*n_j))
    """

    def __init__(self, means, cov, oracle, init_count=1,
                 sigma=1, clipping=True, optimistic=True, path=False,
                 posterior="ellipse",
                 delta=lambda t: max(1, np.log(t)+3*np.log(np.log(t)))):
        if clipping and optimistic:
            super().__init__("Clip Correlated Prior")
        elif optimistic:
            super().__init__("Optimistic Correlated Prior")
        else:
            super().__init__("Correlated Prior")
        self.clipping = clipping
        self.optimistic = optimistic
        self.init_means = means
        self.subg_matrix = cov  # subgaussianity matrix
        self.means = means
        self.cov = np.zeros_like(cov)
        self.oracle = oracle
        self.path = path
        self.init = True  # True if we are still initialization
        self.t = 0  # timestep of algo
        self.posterior = posterior
        if posterior == "app-ellipse":
            self.name += " (app-ellipse)"

        # n_ij is the number of times i,j got played together
        self.N = np.zeros((len(means), len(means)))
        self.delta = delta
        self.sigma = sigma

    def update(self, plays, feedback):
        """
        Update the posterior distribution after playing plays and receiving feedback = X[plays]
        """
        play = np.meshgrid(plays, plays)
        self.N[play] += 1  # update the counters
        N = self.N[play]
        # update empirical mean
        self.means[plays] = (self.means[plays]*(np.diag(self.N[play])-1) +
                             feedback)/(np.diag(self.N[play]))

        # update covariance matrix
        if self.posterior == "ellipse":
            self.cov = self.subg_matrix * \
                (np.diag(1/(np.diag(self.N))) @ (self.N)
                    @ np.diag(1/(np.diag(self.N))))
        elif self.posterior == "app-ellipse":
            self.cov = np.diag(1/(np.sqrt(np.diag(self.N)))) @ \
                self.subg_matrix @ np.diag(1/(np.sqrt(np.diag(self.N))))
        else:
            raise ValueError(
                "Invalid choice of posterior.")

        if self.init:
            # end init. if all arms have been pulled at least init_count times
            self.init = (np.diag(self.N) == 0).any()

    def action(self):
        """
        Return Oracle(theta_t).
        """
        self.t += 1

        if self.init:
            theta = np.zeros_like(self.means)
            pulls = (np.diag(self.N) > 0)
            non_pulls = (np.diag(self.N) == 0)
            if non_pulls.any():
                # uniform distrib for arms not pulled yet
                if self.path:  # means in (-1, 0)
                    theta[non_pulls] = np.random.uniform(
                        -1, 0, size=np.sum(non_pulls))
                else:  # means in (0,1)
                    theta[non_pulls] = np.random.uniform(
                        0, 1, size=np.sum(non_pulls))
            if pulls.any():
                pulls = np.where(pulls)
                theta[pulls] = np.random.multivariate_normal(
                    self.means[pulls], self.cov[np.meshgrid(pulls, pulls)])
                if self.clipping and self.optimistic:
                    bound = self.sigma * \
                        np.sqrt(2*self.delta(self.t)/np.diag(self.N[pulls]))
                    theta[pulls] = np.clip(theta[pulls], self.means[pulls],
                                           self.means[pulls] + bound)
                elif self.optimistic:
                    theta[pulls] = np.max(
                        (theta[pulls], self.means[pulls]), axis=0)
        else:
            theta = np.random.multivariate_normal(self.means, self.cov)
            if self.clipping and self.optimistic:
                theta = np.clip(theta, self.means, self.means + self.sigma *
                                np.sqrt(2*self.delta(self.t)/np.diag(self.N)))
            elif self.optimistic:
                theta = np.max((theta, self.means), axis=0)

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
    CTS-Gaussian implemented for INDEPENDENT prior only (speed up of MultiTS).
    """

    def __init__(self, means, cov, oracle, init_count=1, sigma=1,
                 clipping=True, optimistic=True, path=False,
                 delta=lambda t: max(1, np.log(t)+3*np.log(np.log(t)))):
        if not(np.allclose(np.diag(np.diag(cov)), cov)):
            raise ValueError('The Subgaussian Matrix has to be diagonal')
        if clipping and optimistic:
            super().__init__("Clip CTS-Gaussian")
        elif optimistic:
            super().__init__("Optimistic CTS-Gaussian")
        else:
            super().__init__("CTS-Gaussian")
        self.clipping = clipping
        self.optimistic = optimistic
        self.path = path
        self.init_means = means
        self.means = means
        self.oracle = oracle
        self.init = True  # True if we are still initialization
        self.t = 0  # timestep of algo
        # n_ij is the number of times i,j got played together
        self.N = np.zeros_like(means)
        self.subg_var = np.diag(cov)
        if not(np.all(np.linalg.eigvals(cov) >= -1e-20)):
            raise ValueError("The covariance matrix is not positive.")
        self.cov = np.zeros(len(means))  # diagonal of the covariance of Theta
        self.delta = delta
        self.sigma = sigma

    def update(self, plays, feedback):
        """
        Update the posterior distribution after playing plays and receiving feedback = X[plays]
        """
        self.N[plays] += 1  # update the counters
        # update empirical mean
        self.means[plays] = (
            self.means[plays]*(self.N[plays]-1) + feedback)/(self.N[plays])
        self.cov = self.subg_var/(self.N)  # update variances

        if self.init and not(self.path):
            # end init. if all arms have been pulled at least init_count times
            self.init = (self.N == 0).any()

    def action(self):
        """
        Return Oracle(theta_t).
        """
        self.t += 1

        if self.init:  # uniform for non pulled arms
            theta = np.zeros_like(self.means)
            pulls = (self.N > 0)
            non_pulls = (self.N == 0)
            if non_pulls.any():
                # uniform distrib for arms not pulled yet
                if self.path:  # means in (-1, 0)
                    theta[non_pulls] = np.random.uniform(
                        -1, 0, size=np.sum(non_pulls))
                else:  # means in (0,1)
                    theta[non_pulls] = np.random.uniform(
                        0, 1, size=np.sum(non_pulls))
            if pulls.any():
                theta[pulls] = np.random.multivariate_normal(
                    self.means[pulls], np.diag(self.cov[pulls]))
                if self.clipping and self.optimistic:
                    bound = self.sigma * \
                        np.sqrt(2*self.delta(self.t)/self.N[pulls])
                    theta[pulls] = np.clip(theta[pulls], self.means[pulls],
                                           self.means[pulls] + bound)
                elif self.optimistic:
                    theta[pulls] = np.max(
                        (theta[pulls], self.means[pulls]), axis=0)
        else:
            theta = np.random.multivariate_normal(
                self.means, np.diag(self.cov))  # independent prior here
            if self.clipping and self.optimistic:
                bound = self.sigma*np.sqrt(2*self.delta(self.t)/self.N)
                theta = np.clip(theta, self.means, self.means + bound)
            elif self.optimistic:
                theta = np.max((theta, self.means), axis=0)

        return self.oracle.action(theta)

    def reset(self):
        self.N = np.zeros(len(self.means))
        self.means = np.copy(self.init_means)
        self.cov = np.zeros_like(self.cov)
        self.init = True
        self.t = 0


class CommonPrior(BanditAlgo):
    """
    Implement CTS-Gaussian algorithm with a common prior.
    """

    def __init__(self, means, oracle, var, init_count=1,
                 clipping=False, optimistic=False, path=False,
                 delta=lambda t: max(1, np.log(t)+3*np.log(np.log(t)))):
        if clipping and optimistic:
            super().__init__("Clip Common Prior")
        elif optimistic:
            super().__init__("Optimistic Common Prior")
        else:
            super().__init__("Common Prior")
        self.clipping = clipping
        self.optimistic = optimistic
        self.init_means = means
        self.means = means
        self.oracle = oracle
        self.path = path
        self.init = True  # True if we are still initialization
        self.t = 0  # timestep of algo
        self.N = np.zeros_like(means)
        self.var = var
        self.delta = delta
        self.sigma = np.sqrt(np.max(var))

    def update(self, plays, feedback):
        """
        Update the posterior distribution after playing plays and receiving feedback = X[plays]
        """
        N = self.N[plays]
        self.means[plays] = (self.means[plays]*N + feedback) / \
            (N+1)  # update empirical mean
        self.N[plays] += 1  # update counters

        if self.init:
            # end init. if all arms have been pulled at least init_count times
            self.init = (self.N == 0).any()

    def action(self):
        """
        Return Oracle(theta_t).
        """
        self.t += 1

        if self.init:
            theta = np.zeros_like(self.means)
            pulls = (self.N > 0)
            non_pulls = (self.N == 0)
            if non_pulls.any():
                # uniform distrib for arms not pulled yet
                if self.path:  # means in (-1, 0)
                    theta[non_pulls] = np.random.uniform(
                        -1, 0, size=np.sum(non_pulls))
                else:  # means in (0,1)
                    theta[non_pulls] = np.random.uniform(
                        0, 1, size=np.sum(non_pulls))
            if pulls.any():
                Z = np.random.normal()
                # fully correlated prior
                theta[pulls] = self.means[pulls] + Z * \
                    np.sqrt(self.var[pulls]/self.N[pulls])
                if self.clipping and self.optimistic:
                    bound = self.sigma * \
                        np.sqrt(2*self.delta(self.t)/self.N[pulls])
                    theta[pulls] = np.clip(
                        theta[pulls], self.means[pulls],
                        self.means[pulls] + bound)
                elif self.optimistic:
                    theta[pulls] = np.max(
                        (theta[pulls], self.means[pulls]), axis=0)
        else:
            Z = np.random.normal()
            # fully correlated prior
            theta = self.means + Z*np.sqrt(self.var/self.N)
            if self.clipping and self.optimistic:
                bound = self.sigma*np.sqrt(2*self.delta(self.t)/self.N)
                theta = np.clip(theta, self.means, self.means + bound)
            elif self.optimistic:
                theta = np.max((theta, self.means), axis=0)
        return self.oracle.action(theta)

    def reset(self):
        self.N = np.zeros(len(self.means))
        self.means = np.copy(self.init_means)
        self.init = True
        self.t = 0


# ESCB

class ESCBMatching(BanditAlgo):
    """
    ESCB algo for matching problem when the bipartite graph is a complete graph. 
    Warning ! It has a combinatorial complexity and can not manage large number of arms.
    """

    def __init__(self, g, cov,
                 delta=lambda t: np.maximum(np.log(t) + (len(g.edges)/2 + 2) *
                                            np.log(np.log(t)) +
                                            len(g.edges)/4*np.log(1+np.e), 1)):
        super().__init__("ESCB")
        self.n = len(g.edges)
        self.graph = g.copy()
        self.all_matchings = all_maximal_matchings(g)
        self.left, self.right = nx.bipartite.sets(self.graph)
        self.left, self.right = list(self.left), list(self.right)
        self.m = min(len(self.left), len(self.right))
        self.means = np.zeros(self.n)  # empirical mean
        # n_ij is the number of times that i and j got simultaneously pulled
        self.N = np.zeros((self.n, self.n))
        self.t = 0
        self.delta = delta  # used for the confidence bound
        self.subg_matrix = cov
        if not(np.all(np.linalg.eigvals(cov) >= -1e-7)):
            raise ValueError("The covariance matrix is not positive.")
        self.cov = np.diag(np.inf*np.ones(self.n))
        self.edge_dict = {}
        for i, (u, v) in enumerate(self.graph.edges):
            self.edge_dict[np.min((u, v)), np.max((u, v))] = i

    def matching_to_indices(self, match):
        plays = [self.edge_dict[np.min((u, v)), np.max((u, v))]
                 for u, v in match]
        return np.array(plays)

    def action(self):
        """
        Compute the action brute force. This might take a lot of time, especially when the number of arms is large.
        """
        self.t += 1
        max_val = -np.inf
        max_match = None
        for match in self.all_matchings:
            plays = self.matching_to_indices(match)
            bound = np.sqrt(2*self.delta(self.t) *
                            np.sum(self.cov[np.meshgrid(plays, plays)]))
            val = np.sum(self.means[plays]) + bound
            if val > max_val:
                max_val = val
                max_match = plays
        return max_match

    def update(self, plays, feedback):
        play = np.meshgrid(plays, plays)
        N = self.N[play]
        self.means[plays] = (self.means[plays]*np.diag(N) +
                             feedback)/(np.diag(N)+1)  # update empirical mean
        # update covariance matrix
        self.cov[play] = self.subg_matrix[play] * \
            (np.diag(1/(np.diag(N)+1)) @ (N+1) @ np.diag(1/(np.diag(N)+1)))
        self.N[play] += 1  # update the counters

    def reset(self):
        self.t = 0
        self.means = np.zeros(self.n)
        self.N = np.zeros((self.n, self.n))
        self.cov = np.diag(np.inf*np.ones(self.n))


class ESCBCompleteMatching(BanditAlgo):
    """
    ESCB algo for matching problem when the bipartite graph is a complete graph. 
    Warning ! It has a combinatorial complexity and can not manage large number of arms.
    """

    def __init__(self, g, cov,
                 delta=lambda t: np.maximum(np.log(t) + (len(g.edges)/2 + 2) *
                                            np.log(np.log(t)) +
                                            len(g.edges)/4*np.log(1+np.e), 1)):
        super().__init__("ESCB")
        self.n = len(g.edges)
        self.graph = g.copy()
        self.left, self.right = nx.bipartite.sets(self.graph)
        self.left, self.right = list(self.left), list(self.right)
        self.m = min(len(self.left), len(self.right))
        self.means = np.zeros(self.n)  # empirical mean
        # n_ij is the number of times that i and j got simultaneously pulled
        self.N = np.zeros((self.n, self.n))
        self.t = 0
        self.delta = delta  # used for the confidence bound
        self.subg_matrix = cov
        if not(np.all(np.linalg.eigvals(cov) >= -1e-7)):
            raise ValueError("The covariance matrix is not positive.")
        self.cov = np.diag(np.inf*np.ones(self.n))
        self.edge_dict = {}
        for i, (u, v) in enumerate(self.graph.edges):
            self.edge_dict[np.min((u, v)), np.max((u, v))] = i

    def matching_to_indices(self, match):
        plays = [self.edge_dict[np.min((u, v)), np.max((u, v))]
                 for u, v in match]
        return np.array(plays)

    def action(self):
        """
        Compute the action brute force. This might take a lot of time, especially when the number of arms is large.
        """
        self.t += 1
        max_val = -np.inf
        max_match = None
        for i, right in enumerate(itertools.permutations(self.right)):
            right = list(right)
            match = [(self.left[i], right[i])
                     for i in range(self.m)]  # check all matchings
            plays = self.matching_to_indices(match)
            bound = np.sqrt(2*self.delta(self.t)*np.sum(
                self.cov[np.meshgrid(plays, plays)]))
            val = np.sum(self.means[plays]) + bound
            if val > max_val:
                max_val = val
                max_match = plays
        return max_match

    def update(self, plays, feedback):
        play = np.meshgrid(plays, plays)
        N = self.N[play]
        self.means[plays] = (self.means[plays]*np.diag(N) +
                             feedback)/(np.diag(N)+1)  # update empirical mean
        # update covariance matrix
        self.cov[play] = self.subg_matrix[play] * \
            (np.diag(1/(np.diag(N)+1)) @ (N+1) @ np.diag(1/(np.diag(N)+1)))
        self.N[play] += 1  # update the counters

    def reset(self):
        self.t = 0
        self.means = np.zeros(self.n)
        self.N = np.zeros((self.n, self.n))
        self.cov = np.diag(np.inf*np.ones(self.n))


if __name__ == '__main__':
    pass
