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

![ns_tori](/figures/ns_animation.gif)
