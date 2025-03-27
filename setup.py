"""File to create a pip-intallable package, a scalable solution to access 
resources across the project"""

from setuptools import find_packages, setup

setup(
    name="src",
    packages=find_packages(),
)
