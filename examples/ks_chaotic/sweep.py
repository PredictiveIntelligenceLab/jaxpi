import os

# os.environ["XLA_FLAGS"] = '--xla_gpu_autotune_level=0'
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"  # For better reproducible!  ~35% slower !

from functools import partial

from absl import app
from absl import flags
from absl import logging

import jax

import ml_collections
from ml_collections import config_flags

import wandb

import train

FLAGS = flags.FLAGS

flags.DEFINE_string("workdir", None, "Directory to store model data.")
config_flags.DEFINE_config_file(
    "config",
    None,
    "File path to the training hyperparameter configuration.",
    lock_config=True,
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
        "seed": {"values": [2, 3, 5, 7, 9]},
        "arch_name": {"values": ["Mlp", "ModifiedMlp"]},
        "arch_reparam": {
            "values": [None, {"type": "weight_fact", "mean": 1.0, "stddev": 0.1}]
        },
        "scheme": {"values": ["grad_norm", "ntk"]},
    }
    sweep_config["parameters"] = parameters_dict

    def train_sweep():
        config = FLAGS.config

        wandb.init(project=config.wandb.project, name=config.wandb.name)

        sweep_config = wandb.config

        config.seed = sweep_config.seed
        config.arch.reparam = sweep_config.arch_reparam
        config.arch.arch_name = sweep_config.arch_name
        config.weighting.scheme = sweep_config.scheme

        train.train_and_evaluate(config, workdir)

    sweep_id = wandb.sweep(
        sweep_config,
        project=config.wandb.project,
    )

    wandb.agent(sweep_id, function=train_sweep)


if __name__ == "__main__":
    flags.mark_flags_as_required(["config", "workdir"])
    app.run(main)
