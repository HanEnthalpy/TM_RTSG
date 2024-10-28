from TM_RTSG import TM_RTSG
import numpy as np

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
        if self.f == 1:
            A = np.sum(np.square(x)) / 4000
            B = np.prod(np.cos(x / np.arange(1, len(x) + 1)))
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


f = 1  # function: 1-Griewank
zeta = 1  # std of the noise
num_sample = 2000  # sample budget
d = 5  # dimension of the function
lower_bound = -10  # lower_bound of the domain
upper_bound = 10  # upper_bound of the domain
kernel = 'Laplace'  # kernel: Select from 'Brownian Motion', 'Brownian Bridge', 'Laplace', 'Brownian Field'
parameter = np.array([0.75])  # parameter for the kernel
sigma = 0.01  # regularization parameter


q = func(d, 1, zeta)
s = TM_RTSG(num_sample, d, lower_bound, upper_bound, kernel, parameter, sigma)

s.solve(q.query)  # input data

for i in range(10, 50):
    x = s.sg_design[i, :]  # point to predict
    ans = s.predict(x)
    print("Estimator: %.2f" % ans.value, "MSE: %.2f" % ans.std, "Sample: %.2f" % s.y_sample[i], "Ans: %.2f" % q.query(x, 0))

x = np.random.uniform(lower_bound, upper_bound, s.d)
ans = s.predict(x)
print("Estimator: %.2f" % ans.value, "MSE: %.2f" % ans.std, "Ans: %.2f" % q.query(x, 0))
