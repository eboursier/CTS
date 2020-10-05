from __future__ import print_function
import numpy as np
import os
import networkx as nx


class Reward():
    """
    Superclass for reward functions.
    A Reward must have a self.reward(X, plays) function which returns the reward given by pulling the arm in plays when the mean of the arms is given by X.
    and a self.feedback(X, plays) function to return the semi-bandit feedback reward.
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
        return np.sum(X[plays])  # note that X[plays] is the bandit feedback

    def feedback(self, X, plays):
        """
        Return semi-bandit feedback
        """
        return X[plays]


class MatchingReward(Reward):
    """
    Reward class for matching problem
    """

    def __init__(self, g):
        """
        g is the induced bipartite graph for matchings
        """
        super().__init__("Matching score")
        if not(nx.bipartite.is_bipartite(g)):  # need a bipartite graph
            raise ValueError("The induced graph has to be bipartite.")
        self.graph = g.copy()
        self.ind_dict = {}
        for i, (u, v) in enumerate(self.graph.edges):
            self.ind_dict[i] = (u, v)

    def reward(self, X, plays):
        if not(self.is_matching(plays)):
            raise ValueError('The action has to be a matching')
        return np.sum(X[plays])

    def feedback(self, X, plays):
        return X[plays]

    def plays_to_edges(self, plays):
        """
        return the list of edges given by plays
        """
        return {self.ind_dict[i] for i in plays}

    def is_matching(self, plays):
        """
        check if the pulled edges form a matching in the given bipartite graph
        """
        match = self.plays_to_edges(plays)
        return nx.is_maximal_matching(self.graph, match)  # TODO


class PathReward(Reward):
    """
    Reward class for shortest path problem.
    """

    def __init__(self, graph, source=1, target=33):
        """
        Source and target of the path to compute.
        """
        super().__init__("Path length")
        self.source = source
        self.target = target
        self.graph = graph.copy()
        self.ind_dict = {}
        for i, (u, v) in enumerate(self.graph.edges):
            self.ind_dict[i] = (u, v)

    def reward(self, X, plays):
        """
        g is a weighted graph
        plays is the list of nodes giving the path from source to target.
        """
        if not(self.is_path(plays)):
            raise ValueError('Action has to be a path from {} to {}'.format(
                self.source, self.target))
        return np.sum(X[plays])

    def is_path(self, plays):
        """
        check if the pulled edges form a path from source to target
        """
        edges = [self.ind_dict[p] for p in plays]
        path = [self.source]
        for i in range(len(edges)):
            path.append(self.successor(edges[i], path[i]))
        return (nx.algorithms.simple_paths.is_simple_path(self.graph, path) and
                path[-1] == self.target)

    def feedback(self, X, plays):
        """
        return the list of reward along the path
        """
        return X[plays]

    def successor(self, u, predecessor):
        """
        for an edge (u,v), return the node u or v that is not predecessor. Useful to convert sequence of undirected edges to a path.
        """
        if u[0] == predecessor:
            return u[1]
        elif u[1] == predecessor:
            return u[0]
        else:
            raise ValueError(
                'edge has to link predecessor with some successor')


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

    def __init__(self, m, separate_actions=False):
        """
        separate_actions: if True, any action is of the form {k*m+1, ... ,(k+1)*m}
        m: number of arms per action
        """
        super().__init__("LinearFixedSize")
        self.separate = separate_actions
        self.m = m

    def action(self, X):
        """
        Return the optimal action when the means are given by X.
        """
        if self.separate:
            k = np.argmax([np.sum(X[i*self.m:(i+1)*self.m])
                           for i in range(int(len(X)/self.m))])
            # k is the maximizing bulk
            return np.arange(k*self.m, (k+1)*self.m)
        else:
            # return m maximizing indices of X
            return np.argpartition(X, -self.m)[-self.m:]


class MaxMatchingOracle(Oracle):
    """
    Oracle returning the maximal matching of a bipartite graph.
    """

    def __init__(self, g):
        super().__init__("MaximalMatching")
        if not(nx.bipartite.is_bipartite(g)):  # need a bipartite graph
            raise ValueError("The induced graph has to be bipartite.")
        self.graph = g.copy()
        left, right = nx.bipartite.sets(self.graph)
        self.m = min(len(left), len(right))
        self.edge_dict = {}
        for i, (u, v) in enumerate(self.graph.edges):
            self.edge_dict[np.min((u, v)), np.max((u, v))] = i

    def matching_to_indices(self, match):
        plays = [self.edge_dict[np.min((u, v)), np.max((u, v))]
                 for u, v in match]
        return np.array(plays)

    def action(self, X):
        for i, (u, v) in enumerate(self.graph.edges):
            self.graph[u][v]['weight'] = np.minimum(
                X[i], self.m*100)  # deal with infinite weights
        match = nx.matching.max_weight_matching(
            self.graph, maxcardinality=True)
        return self.matching_to_indices(match)


class ShortestPathOracle(Oracle):
    """
    Oracle returning the shortest path in a weighted graph.
    """

    def __init__(self, graph, source=1, target=33):
        super().__init__("ShortestPath")
        self.source = source
        self.target = target
        self.graph = graph.copy()
        self.edge_dict = {}
        for i, (u, v) in enumerate(self.graph.edges):
            self.edge_dict[np.min((u, v)), np.max((u, v))] = i

    def path_to_indices(self, path):
        """
        Transform a path into a list of indices corresponding the edges indices.
        """
        plays = [self.edge_dict[np.min(
            (path[i], path[i+1])), np.max((path[i], path[i+1]))]
            for i in range(len(path)-1)]
        return np.array(plays)

    def action(self, X):
        for i, (u, v) in enumerate(self.graph.edges):
            # generate graph with weights given by -X
            self.graph[u][v]['weight'] = np.maximum(-X[i], 0)
        path = nx.dijkstra_path(self.graph, self.source, self.target)
        return self.path_to_indices(path)


if __name__ == '__main__':
    pass
