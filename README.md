# PirateNets: Physics-informed Deep Learning with Residual Adaptive Networks

This branch contains the  code for [paper](https://arxiv.org/abs/2402.00326): PirateNets: Physics-informed Deep Learning with Residual Adaptive Networks

![PirateNets](./figures/piratenet.png)

## Usage

The usage instructions for this branch are consistent with those in the main branch. Please refer to the main branch documentation for detailed setup and execution guidelines.

## Benchmarks

The following table shows the performance of PirateNets compared to JAX-PI on a set of benchmark problems. The accuracy is measured in 
relative $L^2$ error between the predicted and true solutions.

| **Benchmark** | PirateNet             | JAX-PI                |     
|---------------|-----------------------|-----------------------|
| Allen-Cahn    | $2.24 \times 10^{−5}$ | $5.37 \times 10^{−5}$ |
| Korteweg–De Vries  | $4.27 \times 10^{−4}$ | $1.96 \times 10^{−3}$ |
| Gray-Scott    | $3.61 \times 10^{−3}$ | $6.13$                |
| Ginzburg-Landau | $1.49 \times 10^{−2}$ | $3.20 \times 10^{−2}$ |
| Lid-driven cavity flow (Re=3200) | $4.21 \times 10^{−2}$ | $1.58 \times 10^{−1}$ |


### Grey-Scott

![Grey-Scott](examples/grey_scott/figures/gs_animation.gif)

### Ginzburg–Landau

![Ginzburg–Landau](examples/ginzburg_landau/figures/gl_animation.gif)



## Citation

    @article{wang2024piratenets,
      title={PirateNets: Physics-informed Deep Learning with Residual Adaptive Networks},
      author={Wang, Sifan and Li, Bowen and Chen, Yuhan and Perdikaris, Paris},
      journal={arXiv preprint arXiv:2402.00326},
      year={2024}
    }