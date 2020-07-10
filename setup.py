#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 19:23:09 2020

@author: Vicky

Python Package Setup
"""
import io
from setuptools import setup
from setuptools import find_packages


with io.open("README.md", "r", encoding="utf-8") as f:
    long_desc = f.read()

with open("requirements.txt", "r") as f:
    install_requires = [x.strip() for x in f.readlines()]


setup(
    name='Neural_PDE',
    version='0.1dev',
    description="Deep learning library for solving partial differential equations",
    author="Vignesh Gopakumar",
    author_email="vignesh7g@gmail.com@gmail.com",
    url = "https://github.com/gitvicky/Neural_PDE",
    packages = find_packages(),
    license='MIT',
    long_description = long_desc,
    long_description_content_type = "text/markdown",
    install_requires = install_requires,

)