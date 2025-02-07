import ml_collections

import jax.numpy as jnp


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    config.mode = "train"

    # Weights & Biases
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = "PINN-KS_chaotic"
    wandb.name = "ActNet_sota"
    wandb.tag = None

    # Set the fractional size of the full temporal domain
    config.time_fraction = 1.0

    # Arch
    config.arch = arch = ml_collections.ConfigDict()
    arch.arch_name = "ActNet"
    arch.embed_dim = 256
    arch.num_layers = 5
    arch.out_dim = 1
    arch.num_freqs = 4
    arch.precision = 'highest'
    arch.periodicity = ml_collections.ConfigDict(
        {"period": (1.0,), "axis": (1,), "trainable": (False,)}
    )

    # Optim
    config.optim = optim = ml_collections.ConfigDict()
    optim.optimizer = "Adam"
    optim.beta1 = 0.9
    optim.beta2 = 0.999
    optim.eps = 1e-8
    optim.learning_rate = 1e-3
    optim.decay_rate = 0.8
    optim.decay_steps = 3500
    optim.grad_accum_steps = 0

    config.training = training = ml_collections.ConfigDict()
    training.max_steps = [250_000, 200_000, 200_000, 150_000, 150_000, 150_000, 150_000, 150_000, 150_000, 150_000]
    training.batch_size = 4096
    training.num_time_windows = 10

    # Weighting
    config.weighting = weighting = ml_collections.ConfigDict()
    weighting.scheme = "grad_norm"
    weighting.init_weights = ml_collections.ConfigDict({"ics": 1000.0, "res": 1.0})
    weighting.momentum = 0.9
    weighting.update_every_steps = 1000

    weighting.use_causal = True
    weighting.causal_tol = 1.0
    weighting.num_chunks = 16

    # Logging
    config.logging = logging = ml_collections.ConfigDict()
    logging.log_every_steps = 100
    logging.log_errors = True
    logging.log_losses = True
    logging.log_weights = True
    logging.log_grads = False
    logging.log_ntk = False
    logging.log_preds = False

    # Saving
    config.saving = saving = ml_collections.ConfigDict()
    saving.save_every_steps = 10000
    saving.num_keep_ckpts = 10

    # # Input shape for initializing Flax models
    config.input_dim = 2

    # Integer for PRNG random seed.
    config.seed = 42

    return config
