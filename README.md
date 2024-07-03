# MNIST Classification with MED and MMD

This repository contains implementations of two classifiers, Minimum Euclidean Distance (MED) and Maximum Mahalanobis Distance (MMD), applied to a subset of the MNIST dataset. The dataset has been reduced to include only two classes (digits 3 and 4) and dimensionality reduction to 2 dimensions has been performed using PCA.


## Files
- `main.ipynb`: Main script to load data, preprocess, train, and evaluate classifiers.
- `MED.py`: Contains the `MED` class implementing the Minimum Euclidean Distance classifier.
- `MMD.py`: Contains the `MMD` class implementing the Maximum Mahalanobis Distance classifier.

## Cloning Repository from GitHub
Use this command to clone the repository from GitHub: <br>
`git clone $REPOSITORY`<br> 
- *(make sure to replace the `$REPOSITORY` with the address of repository)*


## Environment Setup
Please follow these steps to set up the working environment:
1. Install Miniconda (use terminal to execute these commands):
    - Create a Directory:<br>
    `mkdir -p ~/miniconda3`
    - Download the Miniconda Installer:<br>
    `curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -o ~/miniconda3/miniconda.sh`
    - Run the Miniconda Installer:<br>
    `bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3`
    - Remove the Installer Script:<br>
    `rm -rf ~/miniconda3/miniconda.sh`
    - Add to path:<br>
    `source ~/miniconda3/bin/activate`
    - Initialize for bash and zsh shells:<br>
    `~/miniconda3/bin/conda init bash`<br>
    and <br>
    `~/miniconda3/bin/conda init zsh`

    For more information visit [Miniconda](https://docs.anaconda.com/miniconda/) website.

2. Create conda environment and install all packages (I used "data_collection" as a name for my environment): <br>
`conda env create -f environment.yml`

3. Activate the environment: <br>
`conda activate ML`
