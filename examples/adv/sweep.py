import os

# os.environ["XLA_FLAGS"] = '--xla_gpu_autotune_level=0'
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"  # For better reproducible!  ~35% slower !

from absl import app
from absl import flags
from absl import logging

import jax

import ml_collections
from ml_collections import config_flags

import wandb

import train

FLAGS = flags.FLAGS

flags.DEFINE_string("workdir", ".", "Directory to store model data.")
config_flags.DEFINE_config_file(
    "config",
    "./configs/sweep.py",
    "File path to the training hyperparameter configuration.",
)


def main(argv):
    config = FLAGS.config
    workdir = FLAGS.workdir

    sweep_config = {
        "method": "grid",
        "name": "sweep",
        "metric": {"goal": "minimize", "name": "l2_error"},
    }

    parameters_dict = {
        "arch_name": {"values": ["Mlp", "ModifiedMlp"]},
        "layer_size": {"values": [256, 512]},
        "num_layers": {"values": [3, 4, 5]},
        "activation": {"values": ["tanh", "gelu"]},
        "arch_reparam": {
            "values": [
                {"type": "weight_fact", "mean": 0.5, "stddev": 0.1},
                {"type": "weight_fact", "mean": 1.0, "stddev": 0.1},
            ]
        },
        "weighting_scheme": {"values": ["grad_norm", "ntk"]},
        "causal_tol": {"values": [1.0, 10.0]},
    }

    sweep_config["parameters"] = parameters_dict

    def train_sweep():
        config = FLAGS.config

        wandb.init(project=config.wandb.project, name=config.wandb.name)

        sweep_config = wandb.config

        # Update config with sweep parameters
        config.arch.arch_name = sweep_config.arch_name
        config.arch.layer_size = sweep_config.layer_size
        config.arch.num_layers = sweep_config.num_layers
        config.arch.activation = sweep_config.activation
        config.arch.reparam = sweep_config.arch_reparam

        config.weighting.scheme = sweep_config.weighting_scheme
        config.weighting.causal_tol = sweep_config.causal_tol

        train.train_and_evaluate(config, workdir)

    sweep_id = wandb.sweep(
        sweep_config,
        project=config.wandb.project,
    )

    wandb.agent(sweep_id, function=train_sweep)


if __name__ == "__main__":
    flags.mark_flags_as_required(["config", "workdir"])
    app.run(main)
