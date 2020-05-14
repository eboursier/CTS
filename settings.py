from __future__ import print_function
import numpy as np
import os

class Reward():
    """
    Superclass for reward functions.
    A Reward must have a self.reward(X, plays) function which returns the reward given by pulling the arm in plays when the mean of the arms is given by X.
    """
    def __init__(self, function):
        self.function = function


class LinearReward(Reward):
    """
    Linear reward class. Simply returns the sum of all played action stats.
    """

    def __init__(self):
        super().__init__("Linear")

    def reward(self, X, plays):
        """
        return the rewards when the stats of arms are X and the played arms are given by plays.
        plays is the number of each played arm (starting from 0) and is NOT a binary vector
        """
        return np.sum(X[plays]) # note that X[plays] is the bandit feedback



class Oracle():
    """
    Superclass for Oracle function.
    An Oracle must have a self.action(X) function which returns the optimal action when the mean vector is given by X.
    """
    def __init__(self, setting):
        self.setting = setting

class LinearFixedSizeOracle(Oracle):
    """
    Oracle returning the optimal action when the reward is linear and the set of actions is the set of all sets of arms of size m.
    """
    def __init__(self, m):
        super().__init__("LinearFixedSize")
        self.m = m

    def action(self, X):
        """
        Return the optimal action when the means are given by X.
        This corresponds to the indices of the m largest values of X.
        """
        return np.argpartition(X, -self.m)[-self.m:]

if __name__ == '__main__':
    pass