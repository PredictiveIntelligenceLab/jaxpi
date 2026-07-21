from pathlib import Path

from setuptools import find_packages, setup


README = Path(__file__).with_name("README.md").read_text(encoding="utf-8")

setup(
    name="jaxpi",
    version="0.0.1",
    url="https://github.com/PredictiveIntelligenceLab/jaxpi",
    author="Sifan Wang, Shyam Sankaran, Hanwen Wang",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "flax",
        "jax",
        "numpy",
        "optax",
    ],
    extras_require={
        "examples": [
            "absl-py",
            "matplotlib",
            "ml_collections",
            "scipy",
            "tabulate",
            "wandb",
        ],
        "testing": ["pytest"],
    },
    license="Apache 2.0",
    description="A library of PINNs models in JAX Flax.",
    long_description=README,
    long_description_content_type="text/markdown",
)
