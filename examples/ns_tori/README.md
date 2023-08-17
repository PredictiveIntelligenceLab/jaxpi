# Navier–Stokes flow in a 2D torus

## Problem Setup

The partial differential equation is defined as

$$\begin{aligned}
w_t +\mathbf{u} \cdot \nabla w &= \frac{1}{\text{Re}} \Delta w,   \quad \text{ in }  [0, T] \times \Omega,  \\
\nabla \cdot \mathbf{u}  &=0,  \quad \text{ in }  [0, T] \times \Omega, \\
w(0, x, y) &=w_{0}(x, y),   \quad \text{ in }  \Omega,
\end{aligned}$$

For this example, we set Re=100 and aim to simulate the system up to T=10.


## Results

The animation below shows comparison between the exact solution and the prediction.
The model parameter can be found at [google drive link](https://drive.google.com/drive/folders/1n2k2613BWWLcug3CI4i3ZQnBvgrHS1Ph?usp=drive_link).
For a comprehensive log of the loss and weights, please visit [our Weights & Biases dashboard](https://wandb.ai/jaxpi/ns_tori?workspace=user-).


![ns_tori](figures/ns_animation.gif)
