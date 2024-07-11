# MNIST Classification with MED, MMD, KNN, ML, and MAP

This repository contains implementations of several classifiers, including Minimum Euclidean Distance (MED), Maximum Mahalanobis Distance (MMD), k-Nearest Neighbors (KNN), Maximum Likelihood (ML), and Maximum A Posteriori (MAP). These classifiers are applied to a subset of the MNIST dataset. The dataset has been reduced to include only two classes (digits 3 and 4), and dimensionality reduction to 2 dimensions has been performed using PCA. Additionally, a regression version of KNN is implemented and applied to the dataset stored in the "regression_data" folder.

## Project Structure
```
Classic-classifiers/
├── regression_data/           # Directory for datasets for regression
├── classes/                   # Directory containing class implementations
│   ├── __init__.py
│   ├── MED.py                 # Minimum Euclidean Distance (MED) classifier implementation
│   ├── MMD.py                 # Maximum Mahalanobis Distance (MMD) classifier implementation
│   ├── KNN.py                 # k-Nearest Neighbors (KNN) classifier/regression implementation
│   ├── MAP.py                 # Maximum A Posteriori (MAP) classifier implementation
│   ├── ML.py                  # Maximum Likelihood (ML) classifier implementation
├── main.ipynb                 # Notebook demonstrating the use of classifiers
├── knn_regression.ipynb       # Notebook demonstrating the KNN regression
├── .gitignore                 # Git ignore file
├── environment.yml            # Python dependencies
└── README.md                  # Project README file
```


## Cloning Repository from GitHub
Use this command to clone the repository from GitHub: <br>
`git clone git@github.com:lesani-ali/Classic-classifiers.git`<br> 


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

## Acknowledgements
- The MNIST dataset is provided by Yann LeCun and can be found [here](http://yann.lecun.com/exdb/mnist/).
