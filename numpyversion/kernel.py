import numpy as np
import math as ma


class BronianMotion:
    def __init__(self, parameter):
        pass

    def dist(self, x, y):
        return min(x, y)

    def p(self, x):
        return x

    def q(self, x):
        return 1


class BrownianBridge:
    def __init__(self, parameter):
        self.T = parameter[0]

    def dist(self, x, y):
        return min(x, y) * (1 - max(x, y) / self.T)

    def p(self, x):
        return x

    def q(self, x):
        return 1 - x / self.T


class Laplace:
    def __init__(self, parameter):
        self.theta = parameter[0]

    def dist(self, x, y):
        return ma.exp(-self.theta * abs(x - y))

    def p(self, x):
        return ma.exp(self.theta * x)

    def q(self, x):
        return ma.exp(-self.theta * x)


class BrownianField:
    def __init__(self, parameter):
        self.theta = parameter[0]
        self.gamma = parameter[1]

    def dist(self, x, y):
        return self.theta + self.gamma * min(x, y)

    def p(self, x):
        return self.theta + self.gamma * x

    def q(self, x):
        return 1


class kernel:
    def __init__(self, kerneltype, parameter):
        self.s = None
        if kerneltype == 'Brownian Motion':
            self.s = BronianMotion(parameter)
        elif kerneltype == 'Brownian Bridge':
            self.s = BrownianBridge(parameter)
        elif kerneltype == 'Laplace':
            self.s = Laplace(parameter)
        elif kerneltype == 'Brownian Field':
            self.s = BrownianField(parameter)
        else:
            raise ValueError('Undefined Kernel Type')

    def element_dist(self, x, y):
        return self.s.dist(x, y)

    def dist(self, x, y):
        prod = 1
        if type(x) == int or type(x) == float:
            return self.s.dist(x, y)

        if type(x) == np.ndarray:
            d = x.shape[0]
        else:
            d = len(x)
        for i in range(d):
            prod *= self.s.dist(x[i], y[i])
        return prod

    def p(self, x):
        return self.s.p(x)

    def q(self, x):
        return self.s.q(x)
