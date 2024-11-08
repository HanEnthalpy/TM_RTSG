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
        self.Y_sample_mean = None
        self.Y_sample_var = None
        self.K_inv = self.s.K_inv

        self.resample = 0
        self.sigma = np.ones(self.num_sample)

    def solve(self, func, sigma=1, resample=1):
        self.resample = resample
        if type(sigma) == np.ndarray:
            self.sigma = sigma
            if len(sigma) != self.num_sample:
                raise ValueError('Insufficient number of resample. To use variance as sigma, ' +
                                 'the resample should be larger than 1 to estimate the variance')
        if type(sigma) == float:
            self.sigma = self.sigma * sigma
        if type(sigma) == str:
            if sigma != 'var':
                raise ValueError('Invalid input: ' + sigma)
            if (sigma == 'var') and (resample == 1):
                raise ValueError('Insufficient number of resample. To use variance as sigma, ' +
                                 'the resample should be larger than 1 to estimate the variance')
        self.Y_sample_mean = np.zeros(self.num_sample)
        self.Y_sample_var = np.zeros(self.num_sample)
        for i in range(self.num_sample):
            sample_result = np.zeros(self.resample)
            for j in range(self.resample):
                sample_result[j] = func(self.sg_design[i, :])
            self.Y_sample_mean[i] = np.mean(sample_result)
            if type(sigma) == str:
                self.Y_sample_var[i] = np.var(sample_result)
        if type(sigma) == str:
            self.sigma = self.Y_sample_var
        self.s.solve(self.sigma, self.Y_sample_mean)

    def predict(self, x):
        return self.s.predict(x)
