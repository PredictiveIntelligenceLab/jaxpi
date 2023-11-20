import ml_collections

import jax.numpy as jnp


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    config.mode = "train"

    # Weights & Biases
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = "PINN-euler"
    wandb.name = "plain"
    wandb.tag = None

    # Arch
    config.arch = arch = ml_collections.ConfigDict()
    arch.arch_name = "Mlp"
    arch.num_layers = 4
    arch.hidden_dim = 256
    arch.out_dim = 4
    arch.activation = "tanh"
    arch.periodicity = ml_collections.ConfigDict(
        {
            "period": (2 * jnp.pi, 2 * jnp.pi),
            "axis": (1, 2),
            "trainable": (False, False),
        }
    )
    arch.fourier_emb = None
    arch.reparam = None

    # Optim
    config.optim = optim = ml_collections.ConfigDict()
    optim.optimizer = "Adam"
    optim.beta1 = 0.9
    optim.beta2 = 0.999
    optim.eps = 1e-8
    optim.learning_rate = 1e-3
    optim.decay_rate = 0.9
    optim.decay_steps = 2000
    optim.grad_accum_steps = 0

    # Training
    config.training = training = ml_collections.ConfigDict()
    training.max_steps = 100000
    training.batch_size_per_device = 4096

    # Weighting
    config.weighting = weighting = ml_collections.ConfigDict()
    weighting.scheme = None
    weighting.init_weights = ml_collections.ConfigDict(
        {
            "u_ic": 1.0,
            "v_ic": 1.0,
            "rho_ic": 1.0,
            "ru": 1.0,
            "rv": 1.0,
            "rc": 1.0,
            "rd": 1.0,
        }
    )
    weighting.momentum = 0.9
    weighting.update_every_steps = 1000

    weighting.use_causal = False
    weighting.causal_tol = 1.0
    weighting.num_chunks = 16

    # Logging
    config.logging = logging = ml_collections.ConfigDict()
    logging.log_every_steps = 100
    logging.log_errors = True
    logging.log_losses = True
    logging.log_weights = True
    logging.log_preds = False
    logging.log_grads = False
    logging.log_ntk = False

    # Saving
    config.saving = saving = ml_collections.ConfigDict()
    saving.save_every_steps = 10000
    saving.num_keep_ckpts = 10

    # Monitor
    # # Input shape for initializing Flax models
    config.input_dim = 3

    # Integer for PRNG random seed.
    config.seed = 0

    return config
