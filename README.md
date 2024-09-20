# TM_RTSG
Sample and Computationally Efficient Stochastic Kriging in High Dimensions


This project implement the algorithm designed by Liang Ding and Xiaowei Zhang, in the artical [*Sample and Computationally Efficient Stochastic Kriging in High Dimensions*](https://doi.org/10.1287/opre.2022.2367) published in Sep 2022.

The algorithm is designed to predict the value of black-box functions with noise under limited sampling resources.

## Step 1 Download and Import
Download TM_RTSG_.pyd and TM_RTSG.py into your project directory. For import, please use 
```
from TM_RTSG import TM_RTSG
```
## Step 2 Setup
### Instantiate
```
example = TM_RTSG(num_sample, d, lower_bound, upper_bound, kernel='Brownian Field', parameter=np.array([1,1]))
```
#### Paremeters
>__`num_sample`: *int, maximum number of sampling accepted*__

>__`d`: *int, dimension of the function*__

>__`lower_bound`: *float, the lower bound the the domain*__

>__`upper_bound`: *float, the upper bound the the domain*__

>__`kernel`: *{'Brownian Field', 'Brownian Motion', 'Brownian Bridge', 'Laplace'}, kernel for the algorithm, Default = 'Brownian Field'*__
>>*'Brownian Field' :* $k_{BF}(x, y)=\theta+\gamma \min(x, y)$<br>
>>*'Brownian Motion' :* $k_{BM}(x, x)=\min(x, y)$ <br>
>>*'Brownian Bridge' :* $k_{BB}(x, y)=\min(x, y)[1-\frac{\max(x, y)}{T}]$ <br>
>>*'Laplace' :* $k_{Laplace}(x, y)=\exp(-\theta|x-y|)$<br>

>__`parameter`: *np.array, parameters for the kernel, Default = np.array([1, 1])*__
>>The number of parameter is determined by the selected kernel <br>
>>*'Brownian Field' :* $[\theta,  \gamma]$<br>
>>*'Brownian Motion' :* $[]$ <br>
>>*'Brownian Bridge' :* $[T]$ <br>
>>*'Laplace' :* $[\theta]$<br>

#### Attributes
>__`num_sg_sample`: *int, the number of points on the sparse sampling grid*__ <br>
>> Each point will be sampled once, the algorithm will sample for `num_sg_sample` times and `num_sg_sample` $\leq$ `num_sample` <br>

>__`sg_design`: *2d np.array(`num_sg_sample`*$\times$*`d`)*__ <br>
>>`sg_design[i, :]` : the *(i+1)*-th sampling point on the sparse grid

### Set of Parameter
```
example.set_parameter(sigma='Default'):
```
#### Paremeters
> __`sigma`: *1d np.array(length = `num_sg_sample`) or 'Default', regularization parameter*__
>> This parameter matches the diagonal element of the Matrix $\Sigma$ in the article. <br>
>> sigma is automatically set as `np.ones(num_sg_sample)` after instantiation. <br>
>> The length of the array should be equal to the `num_sg_sample` to regularize all the sample points. <br>
>> All the element in the regularization array should be nonzero to guarantee the existence of the inverse. <br>
>> If you choose 'Default', the regularization will be set as `np.ones(num_sg_sample)`. <br>

### Input the Function
```
example.solve(func)
```
#### Paremeters
> __`func`: *function to be estimated*__
>> The function should recieve an 1d numpy array with length = `d` and return a float value.

## Step 3 Prediction
```
ans = s.predict(x)
print("Estimator: %.2f" % ans.value, "MSE: %.2f" % ans.std)
```
#### Input
> __`x`: *1d np.array(length = `d`), the point to be estimated*__ <br>
#### Output
> __`ans`: *result of the prediction*__ <br>
>> `ans.value`: predicted value of func(`x`) <br>
>> `ans.std`: MSE of the estimator <br>

## Example
The example programme can be found in the "Example" directory
### Function with noise
The following class __func__ defined the Griewank and Schwefel-2.22 function with noise. <br>
The returned value is $f(\vec{x}) + \xi$, with $\xi \sim N(0, \zeta^{2}f^{2}(\vec{x}))$ <br>

The following class also support returning the true value of the function by using `query(x, add_std=0)`
```
# func.py
import numpy as np
import math as ma
class func:
    def __init__(self, f=1, zeta=1):
        self.f = f
        self.zeta = zete

    def noise(self, mean, std):
        if std == 0:
            return 0
        return np.random.normal(mean, std)

    def query(self, x, add_std=1):
        result = 0
        x = np.array(x)
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
```
### Use TM_RTSG for prediction
```
# example.py
from TM_RTSG import TM_RTSG
from func import func
import numpy as np

f = 1  # function: 1-Griewank
zeta = 1  # std of the noise
q = func(f, zeta)

num_sample = 2000  # sample budget
d = 10  # dimension of the function
lower_bound = -10  # lower_bound of the domain
upper_bound = 10  # upper_bound of the domain
kernel = 'Brownian Field'  # kernel: Select from 'Brownian Motion', 'Brownian Bridge', 'Laplace', 'Brownian Field'
parameter = np.array([1, 1])  # parameter for the kernel

s = TM_RTSG(num_sample, d, lower_bound, upper_bound, kernel, parameter)

s.solve(q.query)  # input function

x = s.sg_design[1, :]  # point to predict (on the sparse grid)
ans = s.predict(x)
print("Estimator: %.2f" % ans.value, "MSE: %.2f" % ans.std, "Sample: %.2f" % s.y_sample[1], "Ans: %.2f" % q.query(x, 0))

x = np.ones(s.d)  # point to predict (not on the sparse grid)
ans = s.predict(x)
print("Estimator: %.2f" % ans.value, "MSE: %.2f" % ans.std, "Ans: %.2f" % q.query(x, 0))
```




