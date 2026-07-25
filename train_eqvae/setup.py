from setuptools import find_namespace_packages, setup


packages = find_namespace_packages(
    include=[
        "core*",
        "data*",
        "losses*",
        "models*",
        "training*",
    ]
)

setup(
    name="eqvae",
    version="0.0.1",
    description="Flattened EQ-VAE training package",
    packages=packages,
    install_requires=[
        "torch",
        "numpy",
        "tqdm",
        "requests",
    ],
)