# Lid-driven cavity flow

## Problem Set-up

The partial differential equation is defined in a non-dimensional form as

$$\begin{aligned}
    &\mathbf{u} \cdot \nabla \mathbf{u}+\nabla p-\frac{1}{R e} \Delta \mathbf{u}&=0, \quad  (x,y) \in (0,1)^2, \\
    &\nabla \cdot \mathbf{u}&=0, \quad  (x,y) \in (0,1)^2,
\end{aligned}$$

where $\mathbf{u} = (u, v)$ denotes the velocity in $x$ and $y$ directions, respectively, and $p$ is the scalar pressure field. We assume $\mathbf{u}=(1, 0)$ on the top lid of the cavity, and a non-slip boundary condition on the other three walls. We are interested in the velocity and pressure distribution for a Reynolds number of $3200$. 


## Results

### Ablation study

| **Fourier Feature** | **RWF** | **Grad Norm** | **Modified MLP** | **Rel. $L^2$ error** | **Run time (min)** |
|:-------------------:|:-------:|:-------------:|:----------------:|:--------------------:|:------------------:|
|         ✔️         |    ✔️   |       ✔️      |        ✔️       |   $1.34 \times 10^{-1}$  |       58.86       |
|         ❌         |    ✔️   |       ✔️      |        ✔️       |   $7.32 \times 10^{-1}$  |       51.28       |
|         ✔️         |    ❌   |       ✔️      |        ✔️       |   $1.59 \times 10^{-1}$  |       62.01       |
|         ✔️         |    ✔️   |       ❌      |        ✔️       |   $3.38 \times 10^{-1}$  |       57.16       |
|         ✔️         |    ✔️   |       ✔️      |        ❌       |   $5.48 \times 10^{-1}$  |       23.40       |
|         ❌         |    ❌   |       ❌      |        ❌       |   $7.94 \times 10^{-1}$  |       17.96       |


### State of the art

We perform a hyperparameter sweep to find the optimal network architecture, loss weighting scheme and optimizer configuration as 



The best relative $L^2$ error is brought down to $1.58 \times 10^{-1}$. The figure below shows the exact solution, prediction, and absolute error. The model parameter can be found at [google drive link]

<figure>
<img src=figures/ldc_pred.png style="width:100%">
<figcaption align = "center">Figure 1: <em>Lid-driven cavity flow:</em> <em>Left:</em> Predicted velocity of the fine-tuned model. <em>Middle, Right:</em> Comparison of the predicted velocity profiles on the vertical and horizontal center-lines against Ghia <em>et al.</em> The resulting relative L2 error against the reference solution is 1.58e-1.</figcaption>
</figure>