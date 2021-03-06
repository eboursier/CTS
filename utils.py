from __future__ import print_function
import numpy as np
import os
import networkx as nx
import time
import itertools


def potential_paths_from_source_to_target(G, source, target):
    """
    Return only the subgraph with paths from source to target
    """
    G2 = G.copy()
    for (u, v) in G2.edges:
        G2[u][v]['weight'] = 0
    new_edge = True
    while new_edge:
        new_edge = False
        path = nx.dijkstra_path(G2, source, target)
        for i in range(len(path)-1):
            # we visited a new edge
            new_edge = (new_edge or G2[path[i]][path[i+1]]['weight'] == 0)
            G2[path[i]][path[i+1]]['weight'] = 1
    for (u, v) in G.edges:
        if G2[u][v]['weight'] == 0:
            G2.remove_edge(u, v)
    for n in G.nodes:
        if not(nx.has_path(G2, source, n)):
            G2.remove_node(n)
    return G2


def all_maximal_matchings(T):
    """
    Return all maximal-cardinality matchings of a graph
    """
    maximal_matchings = []
    partial_matchings = [{(u, v)} for (u, v) in T.edges()]
    left, right = nx.bipartite.sets(T)
    max_card = min(len(left), len(right))

    while partial_matchings:
        # get current partial matching
        m = partial_matchings.pop()
        nodes_m = set(itertools.chain(*m))

        extended = False
        for (u, v) in T.edges():
            if u not in nodes_m and v not in nodes_m:
                extended = True
                # copy m, extend it and add it to the list of partial matchings
                m_extended = set(m)
                m_extended.add((u, v))
                partial_matchings.append(m_extended)

        if not extended and m not in maximal_matchings:
            maximal_matchings.append(m)

    for s in maximal_matchings:
        if len(s) < max_card:
            maximal_matchings.remove(s)

    return maximal_matchings


def simu(mab, rew, oracle, algo, horizon):
    """
    Simulate an instance for t = 1 , ... , horizon with the setting given by mab (MAB class), rew (Reward class), oracle (Oracle class) and algo (BanditAlgo class).
    Return the evolution of the regret generated by each timestep, as well as the history of pulls of the algo
    """
    pulls = []
    reward = np.zeros(horizon)
    X = mab.simu(steps=horizon)  # generated statistics

    for t in range(horizon):
        plays = algo.action()  # arms pulled
        feedback = rew.feedback(X[t], plays)  # semi bandit feedback
        reward[t] = rew.reward(mab.means, plays)  # pseudo reward
        algo.update(plays, feedback)  # update algo
        pulls.append(plays)

    # the best achievable reward per timestep
    baseline = rew.reward(mab.means, oracle.action(mab.means))
    return (np.cumsum(baseline-reward), pulls)


def runtime(mab, rew, oracle, algo, horizon, initime=100):
    """
    Returns the average time for the algo to compute algo.action() and algo.update().
    """
    X = mab.simu(steps=horizon+initime)

    actiontime = 0.
    updatetime = 0.
    # n is the number of timesteps AFTER initialization

    for t in range(initime):  # do not count initialization steps
        plays = algo.action()  # arms pulled
        feedback = rew.feedback(X[t], plays)  # semi bandit feedback
        algo.update(plays, feedback)  # update algo

    for t in range(initime, initime+horizon):
        _actiontime = time.time()
        plays = algo.action()  # arms pulled
        actiontime += time.time() - _actiontime
        feedback = rew.feedback(X[t], plays)  # semi bandit feedback
        _updatetime = time.time()
        algo.update(plays, feedback)  # update algo
        updatetime += time.time() - _updatetime
        
    return actiontime/horizon, updatetime/horizon


if __name__ == '__main__':
    pass
