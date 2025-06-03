#install with pip install -e .
from setuptools import setup, find_packages

setup(
    name='gene_viz',
    version='0.1.0',
    author='Gene Viz Team',
    packages=find_packages(),  # Automatically find all packages
    install_requires=[],        
)