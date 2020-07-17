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



this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_desc = f.read()

with open(path.join(this_directory, 'requirements.txt'), encoding='utf-8') as f:
    install_requires = f.read()
    
setup(
    name='tf-pde',
    version='0.2dev',
    description="Deep learning library for solving partial differential equations",
    author="Vignesh Gopakumar",
    author_email="vignesh7g@gmail.com",
    url = "https://github.com/gitvicky/tf-pde",
    packages = find_packages(),
    license='MIT',
    long_description = long_desc,
    long_description_content_type = "text/markdown",
    install_requires = install_requires,

)
