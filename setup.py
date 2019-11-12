#!/usr/bin/env python3
"""The setup script."""

from setuptools import setup

reqs=[
    'pandas==0.25.3',
    'librosa==0.7.1'
]

setup(
    name='respiratory-sounds',
    version='0.1dev',
    packages=['respiratory_sounds'],
    license='MIT',
    long_description=open('README.txt').read(),
    install_requires=reqs,
)
