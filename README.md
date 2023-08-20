# JAX-PI

This repository is a comprehensive implementation of physics-informed neural networks (PINNs), seamlessly integrating several advanced network architectures, training algorithms from these papers 

- [Understanding and Mitigating Gradient Flow Pathologies in Physics-Informed Neural Networks](https://epubs.siam.org/doi/10.1137/20M1318043)
- [When and Why PINNs Fail to Train: A Neural Tangent Kernel Perspective](https://www.sciencedirect.com/science/article/pii/S002199912100663X?casa_token=YlzVQK6hGy8AAAAA:bKwMNg70UoeEuisR1cd1KZnR20xspdvYp1dM4jLkl_wfVDX7O1j2IOlGZsYnC4esu7YcMaO_WOIC)
- [Respecting Causality is All You Need for Training Physics-Informed Neural Networks](https://arxiv.org/abs/2203.07404)
- [Random Weight Factorization Improves the Training of Continuous Neural Representations](https://arxiv.org/abs/2210.01274)
- [On the Eigenvector Bias of Fourier Feature Networks: From Regression to Solving Multi-Scale PDEs with Physics-Informed Neural Network](https://www.sciencedirect.com/science/article/abs/pii/S0045782521002759)
- [Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains](https://arxiv.org/abs/2006.10739)
- [A Method for Representing Periodic Functions and Enforcing Exactly Periodic Boundary Conditions with Deep Neural Networks](https://www.sciencedirect.com/science/article/abs/pii/S0021999121001376)
- [Characterizing Possible Failure Modes in Physics-Informed Neural Networks](https://arxiv.org/abs/2109.01050)


This  repository also releases an extensive range of benchmarking examples, showcasing the effectiveness and robustness of our implementation.
Our implementation supports both **single** and **multi-GPU** training, while evaluation is currently limited to
single-GPU setups.

## Installation

Ensure that you have Python 3.8 or later installed on your system.
Our code is GPU-only.
We highly recommend using the most recent versions of JAX and JAX-lib, along with compatible CUDA and cuDNN versions.
The code has been tested and confirmed to work with the following versions:

- JAX 0.4.5
- CUDA 11.7
- cuDNN 8.2


Install JAX-PI with the following commands:

``` 
git clone https://github.com/PredictiveIntelligenceLab/JAX-PI
cd JAX-PI
pip install .
```

## Quickstart

We use [Weights & Biases](https://wandb.ai/site) to log and monitor training metrics. 
Please ensure you have Weights & Biases installed and properly set up with your account before proceeding. 
You can follow the installation guide provided [here](https://docs.wandb.ai/quickstart).

To illustrate how to use our code, we will use the advection equation as an example. 
First, navigate to the advection directory within the `examples` folder:

``` 
cd JAX-PI/examples/advection
``` 
To train the model, run the following command:
```
python3 main.py 
```

Our code automatically supports multi-GPU execution. 
You can specify the GPUs you want to use with the `CUDA_VISIBLE_DEVICES` environment variable. For example, to use the first two GPUs (0 and 1), use the following command:

```
CUDA_VISIBLE_DEVICES=0,1 python3 main.py
```

**Note on Memory Usage**: Different models and examples may require varying amounts of GPU memory. 
If you encounter an out-of-memory error, you can decrease the batch size using the `--config.batch_size_per_device` option.

To evaluate the model's performance, you can switch to evaluation mode with the following command:

```
python main.py --config.mode=eval
```


## Examples

In the following table, we present a comparison of various benchmarks. Each row contains information about the specific benchmark, 
its relative $L^2$ error, and links to the corresponding model [checkpoints](https://drive.google.com/drive/folders/1tc-fASoUmwJTZ4omwsbz1uhdgEGS4z09?usp=drive_link) and Weights & Biases logs. 


| **Benchmark**                          | **Relative $L^2$ Error** |                                                                                     **Checkpoint**                                                                                      | **Weights & Biases** |
|:--------------------------------------:|:------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:--------------------:|
| Allen-Cahn equation                    |  $5.37 \times 10^{-5}$   |                                          [allen_cahn](https://drive.google.com/drive/folders/1MJihlw87l9YiVVLA8JBtCubf8PB6hPZY?usp=drive_link)                                          |  [allen_cahn](https://wandb.ai/jaxpi/allen_cahn?workspace=user-sifanw)  |
| Advection equation                     |  $6.88 \times 10^{-4}$   |                                             [adv](https://drive.google.com/drive/folders/19BEmUYsHWvsj7wgjnpzCxk9NRX70806C?usp=drive_link)                                              |     [adv](https://wandb.ai/jaxpi/adv?workspace=user-sifanw)      |
| Stokes flow                            |  $8.04 \times 10^{-5}$   |                                            [stokes](https://drive.google.com/drive/folders/11T5ht2LGmIZigIKiLvpyUMbSxlrxb1sF?usp=drive_link)                                            |    [stokes](https://wandb.ai/jaxpi/stokes?workspace=user-sifanw)    |
| Kuramoto–Sivashinsky equation          |  $1.61 \times 10^{-1}$   |                                              [ks](https://drive.google.com/drive/folders/1haoDhCUfCq69ptsA2qgiX8yGhwGLbLaT?usp=drive_link)                                              |      [ks](https://wandb.ai/jaxpi/ks?workspace=user-sifanw)      |
| Lid-driven cavity flow                 |  $1.58 \times 10^{-1}$   |                                             [ldc](https://drive.google.com/drive/folders/14bUqullVYHhb68kdwK_lkhzjtvF74nQl?usp=drive_link)                                              |     [ldc](https://wandb.ai/jaxpi/ldc?workspace=user-sifanw)      |
| Navier–Stokes flow in tori             |  $3.53 \times 10^{-1}$   |                                           [ns_tori](https://drive.google.com/drive/folders/1n2k2613BWWLcug3CI4i3ZQnBvgrHS1Ph?usp=drive_link)                                            |     [ns_tori](https://wandb.ai/jaxpi/ns_tori?workspace=user-sifanw)     |
| Navier–Stokes flow around a cylinder   |            -             |                                         [ns_cylinder](https://drive.google.com/drive/folders/1wy_SJUMPOMFM19P9ChGu_cRlk99VRdZ1?usp=drive_link)                                          |     [ns_cylinder](https://wandb.ai/jaxpi/ns_unsteady_cylinder?workspace=user-sifanw)     |


### Decaying Navier-Stokes flow in tori

![ns_tori](examples/ns_tori/figures/ns_animation.gif)

### Vortex shedding
![ns_cylinder](examples/ns_unsteady_cylinder/figures/ns_cylinder_u.gif)

![ns_cylinder](examples/ns_unsteady_cylinder/figures/ns_cylinder_v.gif)

![ns_cylinder](examples/ns_unsteady_cylinder/figures/ns_cylinder_w.gif)


## Citation

    @article{wang2023expert,
      title={An Expert's Guide to Training Physics-informed Neural Networks},
      author={Wang, Sifan and Sankaran, Shyam and Wang, Hanwen and Perdikaris, Paris},
      journal={arXiv preprint arXiv:2308.08468},
      year={2023}
    }




