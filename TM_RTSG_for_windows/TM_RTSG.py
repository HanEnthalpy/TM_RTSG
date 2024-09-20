from TM_RTSG_ import TM_RTSG_
import numpy as np
import warnings

class TM_RTSG:
    def __init__(self, num_sample, d, lower_bound, upper_bound, kernel='Brownian Field', parameter=np.array([1, 1])):
        warnings.filterwarnings('ignore', category=FutureWarning)
        self.num_sample = num_sample
        self.d = d
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.kernel = kernel
        self.parameter = parameter
        if self.upper_bound <= self.lower_bound:
            raise ValueError('Upper bound of the domain should be larger than the lower bound.')
        if self.kernel not in ['Brownian Motion', 'Brownian Bridge', 'Laplace', 'Brownian Field']:
            raise ValueError('Unexpected kernel ' + '"' + kernel + '"')
        para_cnt = {'Brownian Motion': 0, 'Brownian Bridge': 1, 'Laplace': 1, 'Brownian Field': 2}
        if len(parameter) != para_cnt[self.kernel]:
            raise ValueError('Unexpected number of parameters for kernel ' + self.kernel + ', '
                             + str(para_cnt[self.kernel]) + ' required but get ' + str(len(parameter)) + '.')
        self.s = TM_RTSG_()
        self.s.set_parameter(self.num_sample, self.d, self.lower_bound, self.upper_bound, self.kernel, self.parameter)
        self.sg_design = self.s.sg_design
        self.num_sg_sample = self.s.num_sg_sample
        self.sigma = np.ones(self.num_sg_sample)
        self.y_sample = None

    def set_parameter(self, sigma='Default'):
        if sigma == 'Default':
            self.sigma = np.ones(self.num_sg_sample)
            return
        self.sigma = sigma
        if len(self.sigma) != self.num_sg_sample:
            raise ValueError(
                'Length of regularization parameter should be equal to number of points in sparse grid: '
                + str(self.num_sg_sample) + ' required but len(sigma) = ' + str(len(sigma)))

    def solve(self, func):
        self.y_sample = np.zeros(self.num_sg_sample)
        for i in range(self.sg_design.shape[0]):
            self.y_sample[i] = func(self.sg_design[i, :])
        self.s.solve(self.sigma, self.y_sample)

    def predict(self, x):
        return self.s.predict(x)
