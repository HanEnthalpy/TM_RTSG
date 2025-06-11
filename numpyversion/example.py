import numpy as np
from TM_RTSG_KEIBS import TM_RTSG_KEIBS
import warnings
warnings.filterwarnings('ignore')


def f(x):
    return np.sum(np.abs(x)) + np.random.normal()


s = TM_RTSG_KEIBS(num_sample=1000, sg_resample=3, d=2, lb=-3, ub=3, kerneltype='Laplace', parameter=[0.75], tolerance=1e-10)
# Preset Kernel: Brownian Motion, Brownian Bridge, Laplace, Browian Firld
# Support customized kernel by setting kerneltype='customize' and input the kernel through parameter customize_kernel)
s.GenSG(task='Estimate', plot2D=True)
# plot2D: plot the sparsegrid, only valid if d=2

# Verify the correctness
K, InvK = s.Calc_InvK_Classical(s.sample_grid_01)
print(np.max(abs(InvK - s.InvK)))

s.Solve(f, sigma='Unknown')
# sigma: 'Unknown', scalar or ndarray. For 'Unknown' option, the sample std at each point will be used (require sg_resample > 1)

y_pred, pred_std = s.Estimate([2, 2], calc_std=True)
print(y_pred, pred_std)

s = TM_RTSG_KEIBS(num_sample=351*3+100, sg_resample=3, d=3, lb=-3, ub=3, kerneltype='Laplace', parameter=[0.75], tolerance=1e-10)
s.GenSG(task='Optimize')
s.Solve(f, sigma='Unknown', alpha='Unknown')
# alpha is the lambda in the algorithm. For 'Unknown' option, the sample std at each point will be used (require sg_resample > 1)
opt_id, opt_value = s.Optimize(beta=1, minimize=True, plot=True, show_process=True)
print(opt_id, opt_value)