from TM_RTSG import TM_RTSG
import numpy as np
import math as ma
import random


class func:
    def __init__(self, d, f=1, zeta=1):
        self.d = d
        self.f = f
        self.zeta = zeta

    def noise(self, mean, std):
        if std == 0:
            return 0
        return np.random.normal(mean, std)

    def query(self, x, add_std=1):
        result = 0
        x = np.array(x)
        if self.f == 1:  # Griewank
            result = np.sum(np.square(x)) / 4000 - np.prod(np.cos(x)) + 1
        if self.f == 2:  # Schwefel-2.22
            result = np.sum(np.abs(x)) + np.prod(np.abs(x))
        if self.f == 3:  # Rastrigin
            result = 10 * len(x) + np.sum(np.square(x) - 10 * np.cos(x * 2 * ma.pi))
        if self.f == 4:  # Levy
            x = 1 + (x - 1) / 4
            result = (ma.sin(ma.pi * x[0])) ** 2
            result += np.sum((np.square(x[:-1] - 1) * (1 + 10 * np.square(np.sin(ma.pi * x[:-1] + 1)))))
            result += (x[-1] - 1) ** 2 * (1 + (ma.sin(2 * ma.pi * x[-1])) ** 2)
        result += self.noise(0, ma.sqrt(self.zeta) * max(0.01, abs(result))) * add_std
        return result


f_ = 1  # function: 1-Griewank
zeta_ = 10  # std of the noise
num_sample = 1000  # sample budget
d = 20  # dimension of the function
lower_bound = -4  # lower_bound of the domain
upper_bound = 4  # upper_bound of the domain
kernel = 'Laplace'  # kernel: Select from 'Brownian Motion', 'Brownian Bridge', 'Laplace', 'Brownian Field'
parameter = np.array([0.4])  # parameter for the kernel

q = func(d, f_, zeta_)
s = TM_RTSG(num_sample, d, lower_bound, upper_bound, kernel, parameter)
s.solve(q.query, 'var', 10)  # input data

intraerror = []
sample_error = []
samples = random.sample(range(1000), 50)
for i in range(50):
    x = s.sg_design[samples[i], :]  # point to predict
    ans = s.predict(x)
    print("Estimator: %.2f" % ans.value, "MSE: %.2f" % ans.std, "Sample: %.2f" % s.Y_sample_mean[samples[i]],
          "Ans: %.2f" % q.query(x, 0))
    intraerror.append(ans.value - q.query(x, 0))
    sample_error.append(s.Y_sample_mean[samples[i]] - q.query(x, 0))

intererror = []
for i in range(1000):
    x = np.random.uniform(lower_bound, upper_bound, s.d)
    ans = s.predict(x)
    print("Estimator: %.2f" % ans.value, "MSE: %.2f" % ans.std, "Ans: %.2f" % q.query(x, 0))
    intererror.append(ans.value - q.query(x, 0))

print('Root Mean Squared Error for Estimator on the grid: ', ma.sqrt(np.mean(np.square(intraerror))))
print('Root Mean Squared Error for Sampling Mean on the grid: ', ma.sqrt(np.mean(np.square(sample_error))))
print('Root Mean Squared Error for Estimator on random point: ', ma.sqrt(np.mean(np.square(intererror))))
