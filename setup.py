#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="manten",
    version="0.0.1",
    description="manten",
    author="",
    author_email="",
    url="https://github.com/omeryagmurlu/manten",
    install_requires=["lightning", "hydra-core"],
    packages=find_packages(),
    # use this to customize global commands available in the terminal after installing the package
    entry_points={
        "console_scripts": [
            "train_command = manten.train:main",
            "eval_command = manten.eval:main",
        ]
    },
)
