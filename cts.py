from __future__ import print_function
import numpy as np
import os

### The code is really factorized so it is easy to implement new types of reward functions/variable distributions, and so on.
    
class MAB():
    """
    Superclass for different types of MAB setting. 
    A MAB must have a self.simu(steps=n) function which returns n iid samples (X_t)_t where X_t is in R^K. 
    """
    def __init__(self, distrib):
        self.distrib = distrib


class GaussianCombMAB(MAB):
    """
    Bandits with multivariate gaussian reward distributions.
    """

    def __init__(self, means, cov):
        """
        Initialize with the mean and the covariance of the multivariate distribution.
        """
        super().__init__("MultivariateGaussian")
        self.means = means
        self.cov = cov

    def simu(self, steps=1):
        """
        Simulate a variable X ~ N(means, cov)
        """
        return np.random.multivariate_normal(self.means, self.cov, size=steps)


class BinaryCondSumMAB(MAB):
    """
    Variables -X_i are Bernoulli variables conditioned on their sum. 
    It uses a method introduced by Huseby in Exact Sequential Simulation of Binary Variables given their Sum 
    (http://webdoc.sub.gwdg.de/ebook/serien/e/uio_statistical_rr/10-04.pdf)
    """
    def __init__(self, means, s=-1):
        """
        sum is the given fixed value of the sum of X_i. This step requires to compute P(S=s) for any s in 0,n.
        """
        super().__init__("BinaryFixedSum")
        self.unbiased_means = means # E[-X_i]

        if s==-1:
            self.sum = np.sum(means) # the fixed sum is set equal to the expected sum by default
        else:
            self.sum = float(s)

        if not(self.sum.is_integer()):
                raise ValueError('The sum of expectations has to be an integer')

        self.sum = int(self.sum)
        
        if self.sum<0 or self.sum > len(means):
            raise ValueError('The sum has to be fixed between 0 and n')

        self.n = len(means)

        self.S = np.zeros((self.n+1, self.n+1), dtype=float) # S[i,j] = P(sum_{m=i}^n X_m = j) with m going from 0 to n-1 
        self.S[self.n, 0] = 1
        for i in reversed(range(self.n)):
            self.S[i,0] = (1-self.unbiased_means[i])*self.S[i+1,0]                             # case j=m=0
            for j in range(1, self.n+1):
                self.S[i,j] = self.unbiased_means[i]*(self.S[i+1, j-1]-self.S[i+1, j]) + self.S[i+1, j] # eq (2.2) in EXACT SEQUENTIAL SIMULATION OF BINARY VARIABLES GIVEN THEIR SUM

        self.inv_S = np.zeros((self.n+1, self.n+1), dtype=float) # inv_S[i,j] = P(sum_{m=0}^{i-1} X_m = j) needed to compute the biased means
        self.inv_S[0, 0] = 1
        for i in range(1, self.n+1):
            self.inv_S[i,0] = (1-self.unbiased_means[i-1])*self.inv_S[i-1, 0]
            for j in range(1, self.n+1):
                self.inv_S[i,j] = self.unbiased_means[i-1]*self.inv_S[i-1, j-1] + (1-self.unbiased_means[i-1])*(self.inv_S[i-1, j])

        # compute the mean E[X_i | sum X_j = s]
        self.means = np.zeros(self.n) #E[X_i | sum X_j = - s]
        for i in range(self.n):
            r = self.unbiased_means[i]/self.S[0, self.sum]
            for t in range(self.sum):
                self.means[i] -= r*self.S[i+1, t]*self.inv_S[i, self.sum-1-t]

    def simu(self, steps=1):
        """
        Simulate steps instances of X where the sum of X_i is exactly s (or self.sum if not mentioned)
        """
        X = np.zeros((steps, self.n), dtype=int)
        for t in range(steps):
            s_part = 0
            i = 0
            while s_part < self.sum:
                p = self.unbiased_means[i]*self.S[i+1, self.sum-s_part-1]/self.S[i, self.sum-s_part]        # eq (2.3) in EXACT SEQUENTIAL SIMULATION OF BINARY VARIABLES GIVEN THEIR SUM
                X[t, i] = np.random.binomial(n=1, p=p)
                s_part += X[t, i]
                i += 1
        return -X


class BinaryExpSumMAB(MAB):
    """
    Variables -X_i are Bernoulli variables and the sum of their expectation is fixed at the initialization.
    """
    def __init__(self, means, s=-1):
        """
        sum is the given fixed value of the sum of X_i. This step requires to compute P(S=s) for any s in 0,n.
        """
        super().__init__("BinaryFixedSum")
        self.init_means = means # E[-X_i]

        if s==-1:
            self.sum = np.sum(means) # the fixed sum is set equal to the expected sum by default
        else:
            self.sum = float(s)

        if not(self.sum.is_integer()):
                raise ValueError('The sum of expectations has to be an integer')

        self.sum = int(self.sum)
        
        if self.sum<0 or self.sum > len(means):
            raise ValueError('The sum has to be fixed between 0 and n')

        self.n = len(means)

        S = np.zeros((self.n+1, self.n+1), dtype=float) # S[i,j] = P(sum_{m=i}^n X_m = j) with m going from 0 to n-1 
        S[self.n, 0] = 1
        for i in reversed(range(self.n)):
            S[i,0] = (1-self.init_means[i])*S[i+1,0]                             # case j=m=0
            for j in range(1, self.n+1):
                S[i,j] = self.init_means[i]*(S[i+1, j-1]-S[i+1, j]) + S[i+1, j] # eq (2.2) in EXACT SEQUENTIAL SIMULATION OF BINARY VARIABLES GIVEN THEIR SUM

        inv_S = np.zeros((self.n+1, self.n+1), dtype=float) # inv_S[i,j] = P(sum_{m=0}^{i-1} X_m = j) needed to compute the biased means
        inv_S[0, 0] = 1
        for i in range(1, self.n+1):
            inv_S[i,0] = (1-self.init_means[i-1])*inv_S[i-1, 0]
            for j in range(1, self.n+1):
                inv_S[i,j] = self.init_means[i-1]*inv_S[i-1, j-1] + (1-self.init_means[i-1])*(inv_S[i-1, j])

        # compute the mean E[X_i | sum X_j = s]
        self.means = np.zeros(self.n) #E[X_i | sum X_j = - s]
        for i in range(self.n):
            r = self.init_means[i]/S[0, self.sum]
            for t in range(self.sum):
                self.means[i] -= r*S[i+1, t]*inv_S[i, self.sum-1-t]

    def simu(self, steps=1):
        """
        Simulate steps instances of X where the sum of X_i is exactly s (or self.sum if not mentioned)
        """
        return -np.random.binomial(n=1, p=-self.means, size=(steps, self.n))


if __name__ == '__main__':
    pass