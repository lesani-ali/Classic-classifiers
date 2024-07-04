# MNIST Classification with MED, MMD, KNN, ML, and MAP

This repository contains implementations of several classifiers, including Minimum Euclidean Distance (MED), Maximum Mahalanobis Distance (MMD), k-Nearest Neighbors (KNN), Maximum Likelihood (ML), and Maximum A Posteriori (MAP). These classifiers are applied to a subset of the MNIST dataset. The dataset has been reduced to include only two classes (digits 3 and 4), and dimensionality reduction to 2 dimensions has been performed using PCA. Additionally, a regression version of KNN is implemented and applied to the dataset stored in the "regression_data" folder.


## Files
- `MED.py`: Contains the `MED` class implementing the Minimum Euclidean Distance classifier.
- `MMD.py`: Contains the `MMD` class implementing the Maximum Mahalanobis Distance classifier.
- `KNN.py`: Contains the `KNN` class implementing the k-Nearest Neighbors classifier. This also includes the regression version of KNN.
- `ML.py`: Contains the `ML` class implementing the Maximum Likelihood classifier.
- `MAP.py`: Contains the `MAP` class implementing the Maximum A Posteriori classifier.
- `main.ipynb`: Main notebook to load data, preprocess, train, and evaluate classifiers.
- `knn_regression.ipynb`: A notebook to load data, preprocess, train, and evaluate KNN regression model.


## Folders
- "regression_data": Contains dataset for regression.


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
