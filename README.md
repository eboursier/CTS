# CTS
Simulations for "Statistical Efficiency of Thompson Sampling for Combinatorial Semi-Bandits" in collaboration with Pierre Perrault, Michal Valko and Vianney Perchet.

## Required packages
- networkx 2.4
- tqdm

## Run simulations
To run simulations, first create all the repositories folder1/folder2 with
		- folder1 in {simulations, figures}
		- folder2 in {linearfixedsize, maxmatching, shortestpath, shortestpath_corr, linearseparate}

simu.ipynb provides a complete notebook to reproduce all the experiments/figures of the paper

## Organisation of the code
The code is divided in several classes that work independently to allow easy implementation of new settings, distributions or algorithms.

cts.py provides classes to simulate the statistics X for different distributions, as well as a function to process simulations.

settings.py provides classes for reward functions as well as oracle functions, which depend on both the reward function and the action set.

strategies.py provides classes for different algorithms.

utils.py provides some additional useful functions.

## Shortest path dataset
For the shortest path problem, we used the road chesapeake from "The Network Data Repository with Interactive Graph Analytics and Visualization".

## Further info
If you need the exact data generated in the paper or more information, please do not hesitate to send me an email.
