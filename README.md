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
>__num_sample: *int, maximum number of sampling accepted*__

>__d: *int, dimension of the function*__

>__lower_bound: *float, the lower bound the the domain*__

>__upper_bound: *float, the upper bound the the domain*__

>__kernel: *{'Brownian Field', 'Brownian Motion', 'Brownian Bridge', 'Laplace'}, kernel for the algorithm, Default = 'Brownian Field'*__
>>*'Brownian Field' :* $k_{BF}(x, y)=\theta+\gamma \min(x, y)$<br>
>>*'Brownian Motion' :* $k_{BM}(x, x)=\min(x, y)$ <br>
>>*'Brownian Bridge' :* $k_{BB}(x, y)=\min(x, y)[1-\frac{\max(x, y)}{T}]$ <br>
>>*'Laplace' :* $k_{Laplace}(x, y)=\exp(-\theta|x-y|)$<br>

>__parameter: *np.array, parameters for the kernel, Default = np.array([1, 1])*__
>>The number of parameter is determined by the selected kernel <br>
>>*'Brownian Field' :* $[\theta,  \gamma]$<br>
>>*'Brownian Motion' :* $[]$ <br>
>>*'Brownian Bridge' :* $[T]$ <br>
>>*'Laplace' :* $[\theta]$<br>

#### Attributes
>__num_sg_sample: *int, the number of points on the sparse sampling grid*__ <br>
>> Each point will be sampled once, the algorithm will sample for *num_sg_sample* times and *num_sg_sample* $\leq$ *num_sample* <br>

>__sg_design: *2d np.array(num_sg_sample*$\times$*d)*__ <br>
>>*sg_design[i ,  : ] :* the *(i+1)*-th sampling point on the sparse grid

### Set of Parameter
```
example.set_parameter(sigma='Default'):
```
#### Paremeters
> __sigma: *1d np.array(length = num_sg_sample) or 'Default', regularization parameter*__
>> This parameter matches the diagonal element of the Matrix $\Sigma$ in the article. <br>
>> The length of the array should be equal to the num_sg_sample to regularize all the sample points. <br>
>> All the element in the regularization array should be nonzero to guarantee the existence of the inverse. <br>
>> If you choose 'Default', the regularization will be automatically set as np.ones(num_sg_sample). <br>

### Input the Function
```
example.solve(func)
```
#### Paremeters
> __func: *function to be estimated*__
>> The function should recieve an 1d numpy array with length = *d* and return a float value.

## Step 3 Prediction
```
ans = s.predict(x)
print("Estimator: %.2f" % ans.value, "MSE: %.2f" % ans.std)
```
#### Input
> __x: *1d np.array(length = d), the point to be estimated*__ <br>
#### Output
> __ans: *result of the prediction*__ <br>
>> ans.value: predicted value of func(x)
>> ans.std: MSE of the estimator

## Example





