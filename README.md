# Push-and-pull protein dynamics leads to log-normal synaptic sizes and probabilistic multi-spine plasticity

This repository contains the code used to fit the model and generate the figures reported in the relative paper. 
The instructions for running the code and a description of the repository contents are provided below.

# System Requirements

## Hardware Requirements

This package package requires only a standard computer with enough RAM to support the operations defined by a user. For minimal performance, this will be a computer with about 4 GB of RAM.

The runtimes below are generated using a computer with the following specs:
- CPU Intel i9-12900KF, 24 cores, 3.5 MHz
- RAM 64 Gb DDR4
- GPU Nvidia GeForce RTX 3060, VRAM 12 Gb 


## Software Requirements

The package development version is tested on *Linux* operating systems, and in particular on Arch Linux.
The only software requirement for this package is a Python version greater than or equal to 3.11.


# Installing the RDN package
The code in this repository relies on the `rdn` (Reaction-Diffusion Neuron) package contained in the repository
itself. It is, therefore, necessary to install this package before the code can be run.
If you are using a Windows operating system, first install Windows Subsystem for Linux (WSL, https://learn.microsoft.com/en-us/windows/wsl/install) and use the WSL shell to follow the next steps.


1. Clone this repository via
```
$ git clone https://github.com/janko-petkovic/push-pull-code.git
```

2. Move into the folder the repository was cloned in, and create a new virtual environment
```
$ cd <repository-folder-name>
$ python -m venv rdn-venv
```

3. Activate the virtual environment 
```
$ source rdn-venv/bin/activate
```

4. Install the `rdn` package
```
$ pip install .
```

The requirements will be installed automatically.

# Running the code
Once the `rdn` package is installed, you can run the entirety of the code. Here is a description of the available scripts and notebooks, folder by folder.

## `additional analyses`
Contains two notebooks where some complementary analyses on the Chater 2024 and Helm 2022 data is conducted. These analysis have not been reported in the main work, but we include them here for completeness.

## `data`
Folder containing all the data used to fit the model and generate the paper figures.

## `figures-main`
Folder containing the notebooks used to generate the main figures of the paper. Each notebook is named after the figure it generates.
All the runtimes are lower than 1 minute, except for `figure-6.ipynb`, which takes roughly 2 minutes in case the maximum potentiation isosurfaces are recomputed. In the repository, however, we have already included the relative values as a numpy file that is automatically loaded at runtime, located in `figures-main/local-output/change-after-stim.npy`. In case you
want to rerun the computation, just rename or delete this file.

## `figures-supplementary`
Folder containing the notebooks used to generate the supplementary figures of the paper. Each notebook is named after the figure it generates.
All the runtimes are lower than 1 minute, except for `figure-S8-S9-S10-S11.ipynb`, which takes roughly 2 and a half minutes for the sampling.

## `pypesto-fit`
Contains the files and folders used to fit the model to the experimental data and estimate the posterior distributions of the global parameters.
To run these scripts, `cd` into the `pypesto-fit` directory, and call
the script
```
$ cd pypesto-fit
$ python <script-name>.py
```
- `fit-model.py` script used do fit the model to the data. The script automatically loads the optimized model used in this work, but a new fitting
can be imposed setting `force_optimization=True` in the source code. On this machine, the fitting routine takes approximately half an hour.
- `generate-binned-datasets.py` script used to generate the binned data used
for model fitting from the raw data analyzed with the Spyden pipeline.
- `sample-posteriors.py` script used to estimate the posterior distributions of the global parameters. On this machine, the MCMC sampling of 100000 points
takes approximately 10 minutes.

## `scripts`
Python scripts used to generate the dataframes in `raw_data` from the output
of the Spyden pipeline. They cannot be called without the original data (not
included in this repository) and are not supposed to be run.

## `src`
Contains the source code of the `rdn` package, used throughout the entirety
of this repository.

