#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 19:23:09 2020

@author: Vicky

Python Package Setup
"""
import io
from os import path
from setuptools import setup
from setuptools import find_packages

with io.open("README.md", "r", encoding="utf-8") as f:
    long_desc = f.read()

# with open("requirements.txt", "r") as f:
#     install_requires = [x.strip() for x in f.readlines()]
    
setup(
    name='tf-pde',
    version='0.651dev',
    description="Deep learning library for solving partial differential equations",
    author="Vignesh Gopakumar",
    author_email="vignesh7g@gmail.com",
    url = "https://github.com/gitvicky/tf-pde",
    packages = find_packages(),
    license='MIT',
    long_description = long_desc,
    long_description_content_type = "text/markdown",
    install_requires = ['numpy==1.18.5',
                        'matplotlib==3.2.1',
                        'scipy==1.4.1',
                        'sympy==1.6',
                        'tensorflow==2.2.0',
                        'tensorflow-probability==0.10.0',
                        'pydoe==0.3.8',
                        'cloudpickle==1.4.1',
    ],

)
