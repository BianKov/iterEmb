#!/usr/bin/env python3

import os
from setuptools import setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="iteremb",
    version="0.0.0",
    author="BIANKA KOVÁCS",
    license="MIT",
    author_email="kovacs.bianka@ecolres.hu",
    description="Iterative Embedding",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/BianKov/iterEmb",
    packages=["iteremb"],
    install_requires=[line.strip() for line in open("requirements.txt")],
    python_requires=">=3.9",
)
