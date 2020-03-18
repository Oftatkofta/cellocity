# -*- coding: utf-8 -*-

# Learn more: https://github.com/oftatkofta/cellocity

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='Cellocity',
    version='0.0.1',
    description='Velocity and vector analysis of microscopy data',
    long_description=readme,
    author='Jens Eriksson',
    author_email='jens.eriksson@imbim.uu.se',
    url='https://github.com/oftatkofta/cellocity',
    license=license,
    packages=find_packages(exclude=('tests', 'docs', '.idea'))
)