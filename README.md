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
>__`num_sg_sample`: *int, the number of points on the classical sparse sampling grid*__ <br>
>> The algorithm will find the highest level of sparse grid with `num_sg_sample` $\leq$ `num_sample` <br>
>> The remaining `num_sample - num_sg_sample` points will be randomly selected from next level of sparse grid.

>__`sg_design`: *2d np.array(`num_sample`*$\times$*`d`)*__ <br>
>>`sg_design[i, :]` : the *(i+1)*-th sampling point on the sparse grid

>__`K_inv`: *scipy.sparse._csc.csc_matrix*__ <br>
>>The inverse of the kernel matrix by the fast computation algorithm (Algorithm 4)


### Input the Function
```
example.solve(func, sigma=0.1, resample=1)
```
#### Paremeters
> __`func`: *function to be estimated*__ <br>
>> The input of the function an 1d np.array with length = `d`, and return a float value.

> __`sigma`: *float, 1d array or 'var', the regularization parameter, Default = 0.1*__ <br>
>> If the input is a float number, the regularization parameter for all sampling point will be the input float number. <br>
>> If the input is an 1d np.array, the length should be equal to `num_sample`, the regularization parameter will be `sigma[i]` at the i-th sampling point. <br>
>> If the input is 'var', the regularization parameter will be the sample variance at each sampling point. <br>
>> All the element in the regularization array should be nonzero to guarantee the existence of the inverse. <br>

> __`resample`: *int, the number of repeated sampling at each sampling point, Default = 1*__ <br>
>> Note that estimating the sample variance requires `resample` to be larger than 1, if `sigma`='var'.
## Step 3 Prediction
```
ans = s.predict(x)
print("Estimator: %.2f" % ans.value, "MSE: %.2f" % ans.MSE)
```
#### Input
> __`x`: *1d np.array(length = `d`), the point to be estimated*__ <br>
#### Output
> __`ans`: *result of the prediction*__ <br>
>> `ans.value`: predicted value of func(`x`) <br>
>> `ans.MSE`: MSE of the estimator <br>

## Example
The example programme can be found in the "Example" directory
### Function with noise
The following class __func__ defined the Griewank and Schwefel-2.22 function with noise. <br>
The returned value is $f(\vec{x}) + \xi$, with $\xi \sim N(0, \zeta^{2}f^{2}(\vec{x}))$ <br>

The following class also support returning the true value of the function by using `query(x, add_std=0)`
```
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
        if self.f == 5:  # Michalewicz
            result = -np.sum(np.sin(x) * np.power(np.sin(np.square(x) * np.arange(1, len(x)+1) / ma.pi), 20))
        result += self.noise(0, ma.sqrt(self.zeta) * max(0.01, abs(result))) * add_std
        return result
```
### Use TM_RTSG for prediction
> We first try to predict the value on the sparse grid.
>> For this kind of prediction, we can check the sampling value and compare the result.
> Secondly, any points in the domain can be predicted. (The point will be randomly generated with uniform distribution)
```
f_ = 5  # function: 5-Michalewicz
zeta_ = 1  # std of the noise
num_sample = 1000  # sample budget
d = 20  # dimension of the function
lower_bound = 0  # lower_bound of the domain
upper_bound = ma.pi  # upper_bound of the domain
kernel = 'Laplace'  # kernel: Select from 'Brownian Motion', 'Brownian Bridge', 'Laplace', 'Brownian Field'
parameter = np.array([0.2])  # parameter for the kernel

q = func(d, f_, zeta_)
s = TM_RTSG(num_sample, d, lower_bound, upper_bound, kernel, parameter)
s.solve(q.query, 'var', 10)  # input data

intraerror = []
sample_error = []
samples = random.sample(range(num_sample), 50)
for i in range(50):
    x = s.sg_design[samples[i], :]  # point to predict
    ans = s.predict(x)
    print("Estimator: %.2f" % ans.value, "MSE: %.2f" % ans.MSE, "Sample: %.2f" % s.Y_sample_mean[samples[i]],
          "Ans: %.2f" % q.query(x, 0))
    intraerror.append(ans.value - q.query(x, 0))
    sample_error.append(s.Y_sample_mean[samples[i]] - q.query(x, 0))

intererror = []
for i in range(1000):
    x = np.random.uniform(lower_bound, upper_bound, s.d)
    ans = s.predict(x)
    print("Estimator: %.2f" % ans.value, "MSE: %.2f" % ans.MSE, "Ans: %.2f" % q.query(x, 0))
    intererror.append(ans.value - q.query(x, 0))

print('Root Mean Squared Error for Estimator on the grid: ', ma.sqrt(np.mean(np.square(intraerror))))
print('Root Mean Squared Error for Sampling Mean on the grid: ', ma.sqrt(np.mean(np.square(sample_error))))
print('Root Mean Squared Error for Estimator on random point: ', ma.sqrt(np.mean(np.square(intererror))))

```




