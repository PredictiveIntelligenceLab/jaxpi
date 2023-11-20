import ml_collections

import jax.numpy as jnp


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    config.mode = "train"

    # Weights & Biases
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = "PINN-NS_unsteady_cylinder"
    wandb.name = "sota"
    wandb.tag = None

    # Nondimensionalization
    config.nondim = True

    # Arch
    config.arch = arch = ml_collections.ConfigDict()
    arch.arch_name = "ModifiedMlp"
    arch.num_layers = 5
    arch.hidden_dim = 256
    arch.out_dim = 3
    arch.activation = "gelu"  # gelu works better than tanh for this problem
    arch.periodicity = False
    arch.fourier_emb = ml_collections.ConfigDict({"embed_scale": 1.0, "embed_dim": 256})
    arch.reparam = ml_collections.ConfigDict(
        {"type": "weight_fact", "mean": 1.0, "stddev": 0.1}
    )

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
    training.max_steps = 200000
    training.num_time_windows = 10

    training.inflow_batch_size = 2048
    training.outflow_batch_size = 2048
    training.noslip_batch_size = 2048
    training.ic_batch_size = 2048
    training.res_batch_size = 4096

    # Weighting
    config.weighting = weighting = ml_collections.ConfigDict()
    weighting.scheme = "grad_norm"
    weighting.init_weights = {
        "u_ic": 100.0,
        "v_ic": 100.0,
        "p_ic": 100.0,
        "u_in": 100.0,
        "v_in": 100.0,
        "u_out": 1.0,
        "v_out": 1.0,
        "u_noslip": 10.0,
        "v_noslip": 10.0,
        "ru": 1.0,
        "rv": 1.0,
        "rc": 1.0,
    }

    weighting.momentum = 0.9
    weighting.update_every_steps = 1000  # 100 for grad norm and 1000 for ntk

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

    # Input shape for initializing Flax models
    config.input_dim = 3

    # Integer for PRNG random seed.
    config.seed = 42

    return config
