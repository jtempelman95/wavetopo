from setuptools import setup, find_packages

setup(
    name="wavetopo",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
    ],
    extras_require={
        "fenics": ["fenics-dolfinx>=0.7", "mpi4py", "petsc4py"],
    },
)
