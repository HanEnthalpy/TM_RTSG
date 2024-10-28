from TM_RTSG_ import TM_RTSG_
import numpy as np
import warnings


class TM_RTSG:
    def __init__(self, num_sample, d, lower_bound, upper_bound, kernel='Brownian Field', parameter=np.array([1, 1]),
                 sigma=1):
        warnings.filterwarnings('ignore', category=FutureWarning)
        self.num_sample = num_sample
        self.d = d
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.kernel = kernel
        self.parameter = parameter
        self.sigma = np.ones(num_sample)
        if (type(sigma) == float) or (type(sigma) == int):
            self.sigma = self.sigma * sigma
        elif type(sigma) == np.ndarray:
            self.sigma = sigma
            if not (len(sigma) == self.num_sample):
                raise ValueError('Length of sigma is not equal to number of sample.')
        else:
            raise ValueError('Invalid input of sigma')
        if self.upper_bound <= self.lower_bound:
            raise ValueError('Upper bound of the domain should be larger than the lower bound.')
        if self.kernel not in ['Brownian Motion', 'Brownian Bridge', 'Laplace', 'Brownian Field']:
            raise ValueError('Unexpected kernel ' + '"' + kernel + '"')
        para_cnt = {'Brownian Motion': 0, 'Brownian Bridge': 1, 'Laplace': 1, 'Brownian Field': 2}
        if len(parameter) != para_cnt[self.kernel]:
            raise ValueError('Unexpected number of parameters for kernel ' + self.kernel + ', '
                             + str(para_cnt[self.kernel]) + ' required but get ' + str(len(parameter)) + '.')

        self.s = TM_RTSG_()
        self.s.set_parameter(self.num_sample, self.d, self.lower_bound, self.upper_bound, self.kernel, self.parameter,
                          self.sigma)

        self.sg_design = self.s.sg_design
        self.num_sg_sample = self.s.num_sg_sample
        self.y_sample = None
        self.K_inv = self.s.K_inv

    def solve(self, func):
        self.y_sample = np.zeros(self.num_sample)
        for i in range(self.sg_design.shape[0]):
            self.y_sample[i] = func(self.sg_design[i, :])
        self.s.solve(self.y_sample)

    def predict(self, x):
        return self.s.predict(x)
