# CTS
Simulations for "Statistical Efficiency of Thompson Sampling for Combinatorial Semi-Bandits" in collaboration with Pierre Perrault, Michal Valko and Vianney Perchet.

## Organisation of the code
To run simulations, example.ipynb provides a first example of simulation for gaussian noise, linear rewards with fixed size actions. Its goal is mainly to illustrate the code structure and how to use it.

The code is divided in several classes that work independently to allow easy implementation of new settings, distributions or algorithms.

cts.py provides classes to simulate the statistics X for different distributions, as well as a function to process simulations.

settings.py provides classes for reward functions as well as oracle functions, which depend on both the reward function and the action set.

strategies.py provides classes for different algorithms.
