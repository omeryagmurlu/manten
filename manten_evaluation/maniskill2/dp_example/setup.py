from setuptools import find_packages, setup

setup(
    name="diffusion_policy",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["diffusers", "tensorboard", "wandb", "mani_skill"],
    description="A minimal setup for Diffusion Policy for ManiSkill",
)
