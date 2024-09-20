from TM_RTSG import TM_RTSG
from func import func
import numpy as np

f = 1  # function: 1-Griewank
zeta = 1  # std of the noise
wave = 0.1  # wave of the input
q = func(1, zeta, wave)

num_sample = 2000  # sample budget
d = 10  # dimension of the function
lower_bound = -10  # lower_bound of the domain
upper_bound = 10  # upper_bound of the domain
kernel = 'Brownian Field'  # kernel: Select from 'Brownian Motion', 'Brownian Bridge', 'Laplace', 'Brownian Field'
parameter = np.array([1, 1])  # parameter for the kernel

s = TM_RTSG(num_sample, d, lower_bound, upper_bound, kernel, parameter)

s.solve(q.query)  # input data

x = s.sg_design[1, :]  # point to predict
ans = s.predict(x)
print("Estimator: %.2f" % ans.value, "MSE: %.2f" % ans.std, "Sample: %.2f" % s.y_sample[1], "Ans: %.2f" % q.query(x, 0))

x = np.ones(s.d)
ans = s.predict(x)
print("Estimator: %.2f" % ans.value, "MSE: %.2f" % ans.std, "Ans: %.2f" % q.query(x, 0))
