# Advection equation

## Problem Setup

The partial differential equation is defined as 

$$\begin{aligned}
    & \frac{\partial u}{\partial t} + c\frac{\partial u}{\partial x} = 0,\quad t\in[0,1],\ x\in (0, 2\pi),\\
    & u(0,x) = g(x), \quad x\in(0, 2\pi)
\end{aligned}$$

For this example, we set $c=100$ and $g(x) = \sin(x)$.


## Implementation Tips

Our solution assumes periodic behavior over time. 
Similar to spatial periodic boundary conditions, we impose a temporal periodicity. 
Unlike typical boundary conditions, the exact time period here is unknown and treated as a trainable parameter. 
This allows the model to learn and adapt to the most suitable time period. 

Users can enable this feature within the config:

```
arch.periodicity = ml_collections.ConfigDict({
    'period': (2 * jnp.pi, 1.0), 
    'axis': (0, 1), 
    'trainable': (True, False)
})
```

The period tuple sets the periods for the axes, 
axis defines which axes will set periodic embedding, 
and trainable denotes whether the period for each axis is a trainable parameter or a fixed constant.
For further details, see the `PeriodEmbs` class.


## Results

### Ablation study
We conducted an ablation study on **Algorithm 1**, maintaining identical hyperparameters across all tests. The results are displayed in the table below. You can find the specific configuration of the hyperparameters in the `configs` directory. 
For detailed sweep information, please visit [our Weight & Bias dashboard](https://wandb.ai/jaxpi/adv?workspace=user-).


| **Time Period** | **Fourier Feature** | **RWF** | **Grad Norm** | **Causal** | **Rel. $L^2$ Error** | **Runtime (min)** |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ✔️ | ✔️ | ✔️ | ✔️ | ✔️ | $1.02 \times 10^{-2}$ | 9.18 |
| ❌ | ✔️ | ✔️ | ✔️ | ✔️ | $7.37 \times 10^{-1}$ | 8.76 |
| ✔️ | ❌ | ✔️ | ✔️ | ✔️ | $4.29 \times 10^{-1}$ | 7.60 |
| ✔️ | ✔️ | ❌ | ✔️ | ✔️ | $1.31 \times 10^{-2}$ | 9.25 |
| ✔️ | ✔️ | ✔️ | ❌ | ✔️ | $1.13 \times 10^0$ | 7.46 |
| ✔️ | ✔️ | ✔️ | ✔️ | ❌ | $1.49 \times 10^{-2}$ | 9.18 |
| ❌ | ❌ | ❌ | ❌ | ❌ | $9.51 \times 10^{-1}$ | 7.12 |


### State of the art

This section highlights our state-of-the-art results, achieved through an exhaustive hyperparameter sweep. 
This process involved finding the relative optimal combination of network architecture, loss weighting scheme, and optimizer configuration.

To replicate these results, use the following command:

```
python3 main.py --config=configs/sota.py
```

Once training is complete, use the following command to acquire the final predicted error and its corresponding visualization:

```
python3 main.py --config=configs/sota.py --config.mode=eval
```

The best relative $L^2$ error is  $6.884\times 10^{-4}$. The figure below shows the exact solution, prediction, and absolute error. 
The model parameter can be found at [google drive link](https://drive.google.com/drive/folders/19BEmUYsHWvsj7wgjnpzCxk9NRX70806C?usp=drive_link). For a comprehensive log of the loss and weights, please visit [our Weights & Biases dashboard](https://wandb.ai/jaxpi/adv?workspace=user-).

<figure>
<img src=figures/adv_pred.png style="width:100%">
</figure>