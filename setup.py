import os, sys
from setuptools import setup

setup(
    name = "torch_nfp",
    version = "0.0.2dev",
    author = "Yihui Ren",
    author_email = "yren@bnl.gov",
    description = ("An implementation of neural fingerprint"),
    license = "MIT",
    keywords = "example documentation tutorial",
    url = "https://github.com/YHRen/NGFP",
    packages=['NeuralGraph',],
    long_description="",
    install_requires=[
        "torch",
        "scipy",
        "scikit-learn",
        "tqdm",
        "pandas"
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
		"Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
    ],
)
