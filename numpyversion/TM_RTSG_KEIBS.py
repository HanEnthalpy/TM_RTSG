import numpy as np
import math as ma
from SparseGrid import SparseGrid
from kronmult import kronmult
from kernel import kernel
from scipy.special import comb
from scipy.sparse import eye, csr_matrix, csc_matrix, coo_matrix, lil_matrix, diags
import scipy.sparse.linalg as spla
from scipy.stats import norm
import matplotlib.pyplot as plt
import warnings
import time


def eta(x):
    return x * norm.cdf(x) + norm.pdf(x)


class TM_RTSG_KEIBS:
    def __init__(self, num_sample: int, sg_resample: int, d: int, lb, ub, kerneltype: str = 'Brownian Field',
                 parameter=None, customize_kernel=None, tolerance=1e-10):
        if parameter is None:
            parameter = [1, 1]
        self.alpha = None
        self.beta = None

        self.num_sample = num_sample
        self.sg_resample = sg_resample
        self.d = d
        self.lb = lb
        self.ub = ub
        self.tolerance = tolerance
        if kerneltype in ['Brownian Motion', 'Brownian Bridge', 'Laplace', 'Brownian Field']:
            self.kernel = kernel(kerneltype, parameter)
        elif kerneltype == 'Customize':
            self.kernel = customize_kernel
        else:
            raise ValueError(f"Invalid type of kernel. Expect 'Brownian Motion', 'Brownian Bridge', 'Laplace', "
                             f"'Brownian Field' or 'Customize' but receive {kerneltype}.")

        self.task = None
        self.n = None
        self.LTblock = None

        self.num_sg_sample = None
        self.num_base_sg_sample = None

        self.sample_grid_01 = None
        self.sample_grid = None
        self.K = None
        self.InvK = None

        temp_n = self.d + 1
        block1 = 0
        while True:
            self.LTblock = block1
            block1 = 0
            for i in range(self.d, temp_n):
                block1 += pow(2, i - self.d) * int(comb(i - 1, self.d - 1))
            num_sg_sample = block1 + pow(2, temp_n - self.d) * int(comb(temp_n - 1, self.d - 1))
            if num_sg_sample < self.num_sample // self.sg_resample:
                temp_n += 1
            else:
                break

        self.n = temp_n - 1
        self.sample_mean = None
        self.sample_std = None
        self.sample_size = None
        self.sample = None
        self.sigma = None
        self.InvSigma = None
        self.InvSigmaY = None
        self.InvK_InvSigma = None
        self.solver = None
        self.solveresult = None
        self.func = None
        self.max_record = []
        self.sample_path = []
        self.num_sequential_sample = None

    def fowardmap(self, x):
        return x * (self.ub - self.lb) + self.lb

    def backwardmap(self, x):
        return (x - self.lb) / (self.ub - self.lb)

    def GenSG(self, task: str = 'Estimate', plot2D: bool = True):
        if task not in ['Estimate', 'Optimize']:
            raise ValueError(f"Invalid type of task. Expect 'Estimate' or 'Optimize' but receive {task}")
        self.task = task
        if task == 'Estimate':
            sg_base = SparseGrid(self.n, self.d)
            sg_ext = SparseGrid(self.n + 1, self.d)
            self.num_base_sg_sample = sg_base.design.shape[0]
            self.num_sg_sample = self.num_sample // self.sg_resample
            idx = np.random.choice(np.arange(self.num_base_sg_sample, sg_ext.design.shape[0]),
                                   size=self.num_sg_sample - self.num_base_sg_sample, replace=False)
            self.sample_grid_01 = np.empty([self.num_sg_sample, self.d])
            self.sample_grid_01[:self.num_base_sg_sample, :] = sg_base.design
            self.sample_grid_01[self.num_base_sg_sample:, :] = sg_ext.design[idx, :]
            self.sample_grid = self.fowardmap(self.sample_grid_01)

            InvA = self.Calc_InvK(sg_base)
            KN = np.empty([self.num_base_sg_sample, self.num_sg_sample - self.num_base_sg_sample])
            Dv = np.ones(self.num_sg_sample - self.num_base_sg_sample)

            for i in range(self.num_base_sg_sample):
                for j in range(self.num_sg_sample - self.num_base_sg_sample):
                    KN[i, j] = self.kernel.dist(self.sample_grid_01[i, :],
                                                self.sample_grid_01[j + self.num_base_sg_sample, :])

            B = InvA @ KN
            mask = np.abs(B) >= self.tolerance
            rows, cols = np.where(mask)
            values = B[mask]
            B = csc_matrix((values, (rows, cols)), shape=B.shape)

            for i in range(self.num_sg_sample - self.num_base_sg_sample):
                for j in range(self.d):
                    v = self.sample_grid_01[self.num_base_sg_sample + i, j]
                    p = self.kernel.p(v)
                    q = self.kernel.q(v)
                    powerlevel = 0
                    while ma.floor(v) != v:
                        v = v * 2
                        powerlevel += 1
                    power = ma.pow(2, powerlevel)

                    if ma.floor(v) == 1:
                        p_1 = 0
                        q_1 = 1
                    else:
                        p_1 = self.kernel.p((v - 1) / power)
                        q_1 = self.kernel.q((v - 1) / power)
                    if ma.floor(v) == power - 1:
                        p1 = 1
                        q1 = 0
                    else:
                        p1 = self.kernel.p((v + 1) / power)
                        q1 = self.kernel.q((v + 1) / power)
                    Dv[i] *= (p1 * q_1 - p_1 * q1) / ((p * q_1 - p_1 * q) * (p1 * q - p * q1))
            D = diags(Dv).tocsc()
            BD = B * D
            BDBT = BD @ B.T
            self.InvK = lil_matrix((self.num_sg_sample, self.num_sg_sample))
            self.InvK[:self.num_base_sg_sample, :self.num_base_sg_sample] = InvA + BDBT
            self.InvK[self.num_base_sg_sample:, :self.num_base_sg_sample] = -BD.T
            self.InvK[:self.num_base_sg_sample, self.num_base_sg_sample:] = -BD
            self.InvK[self.num_base_sg_sample:, self.num_base_sg_sample:] = D
            self.InvK = self.InvK.tocsc()

        if task == 'Optimize':
            sg = SparseGrid(self.n, self.d)
            self.num_base_sg_sample = sg.design.shape[0]
            self.num_sg_sample = sg.design.shape[0]
            self.sample_grid_01 = sg.design
            self.sample_grid = self.fowardmap(sg.design)
            self.InvK = self.Calc_InvK(sg)
            self.num_sequential_sample = self.num_sample - self.num_sg_sample * self.sg_resample

        self.sample = [[] for _ in range(self.num_sg_sample)]

        if plot2D and self.d == 2:
            plt.scatter(self.sample_grid[:self.num_base_sg_sample, 0], self.sample_grid[:self.num_base_sg_sample, 1],
                        color='blue', s=5)
            plt.scatter(self.sample_grid[self.num_base_sg_sample:, 0], self.sample_grid[self.num_base_sg_sample:, 1],
                        color='red', s=5)
            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2')
            plt.show()

    def Solve(self, func, sigma='Unknown', alpha='Unknown'):
        self.func = func
        if (sigma == 'Unknown') and (self.sg_resample == 1):
            raise ValueError('For unknown standard deviation, at least 2 samples at each point are required.')
        self.sample_mean = np.empty(self.num_sg_sample)
        self.sample_std = np.empty(self.num_sg_sample)
        self.sample_size = np.full(self.num_sg_sample, self.sg_resample)
        for i in range(self.num_sg_sample):
            for j in range(self.sg_resample):
                sample = self.func(self.sample_grid[i, :])
                self.sample[i].append(sample)
            self.sample_mean[i] = np.mean(self.sample[i])
            self.sample_std[i] = np.std(self.sample[i])
            self.sample_std = np.clip(self.sample_std, self.tolerance, None)
        if sigma == 'Unknown':
            self.sigma = self.sample_std
        elif np.isscalar(sigma):
            self.sigma = np.full(self.num_sg_sample, sigma)
        else:
            self.sigma = np.array(sigma)

        if self.task == 'Optimize':
            if alpha == 'Unknown':
                self.alpha = self.sample_std
            elif np.isscalar(alpha):
                self.alpha = np.full(self.num_sg_sample, alpha)
            else:
                self.alpha = np.array(alpha)

        if self.task == 'Estimate':
            self.InvSigma = np.diag(1 / self.sigma)
            self.InvSigmaY = self.InvSigma @ self.sample_mean
            self.InvK_InvSigma = self.InvK + diags(1 / self.sigma)
            self.solver = spla.splu(self.InvK_InvSigma)
            self.solveresult = self.InvSigma @ self.solver.solve(self.InvSigmaY)
        else:
            self.InvSigma = np.diag(1 / self.alpha)
            self.InvK_InvSigma = self.InvK + diags(1 / self.alpha)
            self.solver = spla.splu(self.InvK_InvSigma)

        if sigma == 'Unknown':
            self.sigma = sigma

    def Estimate(self, x: np.ndarray, calc_std: bool = True):
        x = np.array(x)
        if x.shape[0] != self.d:
            raise ValueError("Invalid input for estimation.")
        x_01 = self.backwardmap(x)
        k = np.empty(self.num_sg_sample)
        for i in range(self.num_sg_sample):
            k[i] = self.kernel.dist(x_01, self.sample_grid_01[i, :])
        pred = k.T @ self.InvSigmaY - k.T @ self.solveresult
        if not calc_std:
            return pred
        InvSigmak = self.InvSigma @ k
        mse = self.kernel.dist(x_01, x_01) - k.T @ InvSigmak + InvSigmak.T @ self.solver.solve(InvSigmak)
        return pred, ma.sqrt(mse)

    def Optimize(self, beta=1, minimize: bool = False, plot: bool = True, show_process: bool = True):
        y_pred = np.empty(self.num_sg_sample)
        std_pred = np.empty(self.num_sg_sample)
        if np.isscalar(beta):
            self.beta = np.full(self.num_sequential_sample, beta)
        else:
            self.beta = np.array(beta)
            if self.beta.shape[0] < self.num_sequential_sample:
                raise ValueError(
                    f"The length of beta is smaller than the number of sequential sampling ({self.num_sequential_sample}.")

        if minimize:
            self.sample_mean = -self.sample_mean

        self.InvSigmaY = self.InvSigma @ self.sample_mean
        self.solveresult = self.InvSigma @ self.solver.solve(self.InvSigmaY)

        for i in range(self.num_sg_sample):
            y_pred[i], std_pred[i] = self.Estimate(self.sample_grid[i, :], calc_std=True)

        start = time.perf_counter()
        for i in range(self.num_sequential_sample):
            max_idx = np.argmax(y_pred)
            max_value = y_pred[max_idx]
            self.max_record.append(max_value)
            sigma = self.sigma
            if sigma == 'Unknown':
                sigma = self.sample_std
            score = std_pred * eta((y_pred + self.beta[i] * sigma / np.sqrt(self.sample_size) - max_value) / std_pred)
            sample_idx = np.argmax(score)
            self.sample_path.append(sample_idx)
            sample = self.func(self.sample_grid[sample_idx])
            self.sample[sample_idx].append(sample)
            if minimize:
                sample = -sample
            self.sample_mean[sample_idx] = (self.sample_mean[sample_idx] * self.sample_size[sample_idx] + sample) / (
                    self.sample_size[sample_idx] + 1)
            self.sample_std[sample_idx] = max(np.std(self.sample[sample_idx]), self.tolerance)
            self.sample_size[sample_idx] += 1

            self.InvSigmaY = self.InvSigma @ self.sample_mean
            self.solveresult = self.InvSigma @ self.solver.solve(self.InvSigmaY)

            for j in range(self.num_sg_sample):
                y_pred[j] = self.Estimate(self.sample_grid[j, :], calc_std=False)

            if show_process:
                process = 80
                current = i * process // self.num_sequential_sample
                finish = "▓" * current
                need_do = "-" * (process - 1 - current)
                dur = time.perf_counter() - start
                value = max_value
                if minimize:
                    value = -value
                print("\r{}/{} {:^3.0f}% [{}{}]{:.2f}s, optimal_value={:.4f}".format(i + 1, self.num_sequential_sample,
                      (i + 1) / self.num_sequential_sample * 100, finish, need_do, dur, value), end="")
        if show_process:
            print()
        max_idx = np.argmax(y_pred)
        self.sample_path = np.array(self.sample_path)
        self.max_record = np.array(self.max_record)
        if minimize:
            self.max_record = -self.max_record
            self.sample_mean = - self.sample_mean
            if plot:
                plt.plot(self.max_record)
                plt.show()
            return self.sample_grid[max_idx], -y_pred[max_idx]

        if plot:
            plt.plot(self.max_record)
            plt.show()
        return self.sample_grid[max_idx], y_pred[max_idx]

    def Calc_InvK(self, sg):
        D = []
        for i in range(sg.n - sg.d + 1):
            Dist = np.zeros([len(sg.grids[i][0]), len(sg.grids[i][0])])
            for r in range(len(sg.grids[i][0])):
                for c in range(len(sg.grids[i][0])):
                    Dist[r, c] = self.kernel.dist(sg.grids[i][0][r], sg.grids[i][0][c])
            Dist = np.linalg.inv(Dist)
            mask = np.abs(Dist) >= self.tolerance
            rows, cols = np.where(mask)
            values = Dist[mask]
            D.append(csr_matrix((values, (rows, cols)), shape=Dist.shape))

        l = len(sg.labels)
        InvK = lil_matrix((sg.design.shape[0], sg.design.shape[0]))

        for i in range(l):
            idxs = sg.levelsets[i, :]
            temp = [D[_] for _ in idxs]
            dims = 2 ** (idxs + 1) - 1
            I_dim = np.prod(dims)
            I = eye(I_dim).tocsr()
            T_temp = kronmult(temp, I)
            indices = sg.labels[i]
            InvK[np.ix_(indices, indices)] += sg.coeff[i] * T_temp

        return InvK.tocsc()

    def Calc_InvK_Classical(self, sample_grid):
        K = np.zeros([sample_grid.shape[0], sample_grid.shape[0]])
        for i in range(sample_grid.shape[0]):
            for j in range(sample_grid.shape[0]):
                K[i, j] = self.kernel.dist(sample_grid[i, :], sample_grid[j, :])
        InvK = np.linalg.inv(K)
        return K, csc_matrix(InvK)






