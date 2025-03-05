# -*- coding: utf-8 -*-
# Learn more: https://github.com/oftatkofta/cellocity

from setuptools import setup, find_packages


with open('readme.md') as f:
    readme = f.read()

setup(
    name="cellocity",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'tifffile',
        'matplotlib'
    ],
    description='Velocity and vector analysis of microscopy data',
    long_description=readme,
    long_description_content_type='text/markdown',
    author='Jens Eriksson',
    author_email='jens.eriksson@imbim.uu.se',
    url='https://github.com/oftatkofta/cellocity',
    license="GPLv3",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Topic :: Scientific/Engineering",
    ],
    python_requires='>=3.6',
    project_urls={
        'Documentation': 'https://cellocity.readthedocs.io/en/latest/',
        'Research group': 'https://www.imbim.uu.se/research-groups/infection-and-defence/sellin-mikael/',
        'Source': 'https://github.com/oftatkofta/cellocity',
    },
    entry_points={
        'console_scripts': [
            'cellocity-gui=cellocity.gui:main',
        ],
    },
)