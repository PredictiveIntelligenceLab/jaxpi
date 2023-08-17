# Kuramoto–Sivashinsky equation

## Problem Set-up

The partial differential equation is defined as

$$\begin{aligned}
    &u_{t}+ \alpha u u_x + \beta  u_{x x}+ \gamma u_{x x x x}=0, \quad t \in [0, 1], \ x \in [0, 2 \pi],\\
    &u(0, x) = u_0(x).
\end{aligned}$$

Specifically, we take $\alpha = 100/16, \beta=100/16^2,  \gamma=100/16^4$ and $u_0(x) = \cos(x)(1 + \sin(x))$. 

## Implementation tips

- **Use Time-marching strategy:** Use this technique to handle long-time integration of complex problems
- **Expand Temporal Domain:** For each time window, slightly extend the temporal domain to cover the endpoint, at which the prediction serves as the initial condition for the next time window.
- **Scale input coordinates:** Scale the input coordinates into the range of $[0, 1]$ during the network forward pass.


## Results
### Ablation study

We perform an ablation study on **Algorithm 1**. Table below shows the result of the ablation study. The detail sweep can be found at [weight and bias link](https://wandb.ai/jaxpi/ks?workspace=user-)


| **Modified MLP** | **Fourier Feature** | **RWF** | **Grad Norm** | **Causal** | **Rel. $L^2$ Error** | **Runtime (min)** |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ✔️ | ✔️ | ✔️ | ✔️ | ✔️ | **$1.42 \times 10^{-4}$** | 13.33 |
| ❌ | ✔️ | ✔️ | ✔️ | ✔️ | $2.98 \times 10^{-3}$ | 6.21 |
| ✔️ | ❌ | ✔️ | ✔️ | ✔️ | $1.86 \times 10^{-2}$ | 7.60 |
| ✔️ | ✔️ | ❌ | ✔️ | ✔️ | $1.86 \times 10^{-4}$ | 14.11 |
| ✔️ | ✔️ | ✔️ | ❌ | ✔️ | $2.19 \times 10^{-1}$ | 14.11 |
| ✔️ | ✔️ | ✔️ | ✔️ | ❌ | $2.58 \times 10^{-4}$ | 9.18 |
| ❌ | ❌ | ❌ | ❌ | ❌ | $2.59 \times 10^{-1}$ | 7.12 |


### State of the art


The best relative $L^2$ error is brought down to $1.61 \times 10^{-1}$. The figure below shows the exact solution, prediction, and absolute error. The model parameter can be found at [google drive link](https://drive.google.com/drive/folders/1haoDhCUfCq69ptsA2qgiX8yGhwGLbLaT?usp=drive_link)

<figure>
<img src=figures/ks_chaotic.png style="width:100%">
</figure>