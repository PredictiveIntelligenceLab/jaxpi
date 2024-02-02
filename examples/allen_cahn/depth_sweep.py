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
    "./configs/depth_sweep.py",
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
        "arch_name": {"values": ["PirateNet", "ModifiedMlp"]},
        "seed": {"values": [2, 3, 5, 7, 11]},
        "num_layers": {"values": [1, 2, 3, 4]},
    }

    sweep_config["parameters"] = parameters_dict

    def train_sweep():
        config = FLAGS.config

        wandb.init(project=config.wandb.project, name=config.wandb.name)

        sweep_config = wandb.config

        # Update config with sweep parameters
        config.arch.arch_name = sweep_config.arch_name

        # Since each bottleneck block contains 3 layers
        if config.arch.arch_name == "ModifiedMlp":
            config.arch.num_layers = sweep_config.num_layers * 3
            config.use_pi_init = False
        elif config.arch.arch_name == "PirateNet":
            config.arch.num_layers = sweep_config.num_layers
            config.use_pi_init = True

        config.seed = sweep_config.seed
        config.arch.pi_init = (
            None  # Reset pi_init every sweep otherwise it will be overwritten!!!
        )

        train.train_and_evaluate(config, workdir)

    sweep_id = wandb.sweep(
        sweep_config,
        project=config.wandb.project,
    )

    wandb.agent(sweep_id, function=train_sweep)


if __name__ == "__main__":
    flags.mark_flags_as_required(["config", "workdir"])
    app.run(main)
