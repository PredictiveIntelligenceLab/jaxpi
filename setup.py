from setuptools import setup, find_packages
import os

_CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))

try:
    README = open(os.path.join(_CURRENT_DIR, "README.md"), encoding="utf-8").read()
except IOError:
    README = ""

setup(
    name="jaxpi",
    version="0.0.1",
    url="https://github.com/PredictiveIntelligenceLab/JAX-PI",
    author="Sifan Wang",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "absl-py",
        "flax",
        "jax",
        "jaxlib",
        "matplotlib",
        "ml_collections",
        "numpy",
        "optax",
        "scipy",
        "wandb",
    ],
    extras_require={
        "testing": ["pytest"],
    },
    license="Apache 2.0",
    description="A library of PINNs models in JAX Flax.",
    long_description=open(os.path.join(_CURRENT_DIR, "README.md")).read(),
    long_description_content_type="text/markdown",
)
