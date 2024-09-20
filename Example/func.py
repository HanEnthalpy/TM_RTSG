import numpy as np
import math as ma


class func:
    def __init__(self, f=1, zeta=1, wave=1):
        self.f = f
        self.zeta = zeta
        self.wave = wave

    def noise(self, mean, std):
        if std == 0:
            return 0
        return np.random.normal(mean, std)

    def query(self, x, add_std=1):
        result = 0
        x = np.array(x)
        for i in range(len(x)):
            x[i] += self.noise(0, self.wave*add_std)
        if self.f == 1:
            A = 0
            B = 1
            for i in range(len(x)):
                A += x[i] ** 2 / 4000
                B *= ma.cos(x[i] / (i + 1))
            result = - 50 * (A - B + 1)
        if self.f == 2:
            A = 0
            B = 1
            for i in range(len(x)):
                A += abs(x[i])
                B *= abs(x[i])
            result = - A - B + 100
        result += self.noise(0, self.zeta * min(1, abs(result))) * add_std
        return result
