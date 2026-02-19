# Paper_OGLE

This repository contains the code used in the work described in the paper:

**Paper:** [Astronomy & Astrophysics (2024)](https://www.aanda.org/articles/aa/full_html/2024/11/aa49995-24/aa49995-24.html)

## Installation

To use the code from this project, clone the repository:

git clone https://github.com/Monsalves-Gonzalez-N/Paper_OGLE.git

## Environment setup

Create a conda environment with the required packages. The repository includes a `.yml` file containing all dependencies used in this work.

**Recommended:** use **micromamba**, since it is faster than standard conda.

micromamba create -n CNN_OGLE_legacy -f 2dhistogramCNN.yml

## Usage

The notebook **`How to use the CNN.ipynb`** provides examples of how to use the CNN:

- With a single light curve (LC)
- With an array of light curves (LCs)

