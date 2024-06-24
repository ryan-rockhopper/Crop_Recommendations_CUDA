# CUDA Learning Project

## Author: Ryan Dalrymple

## Start date: May 16 2024

## Pause date: June 24 2024

## End date: TBA

### Overview
The purpose of this project is for me to learn and develop my skills with various libraries. I will be mainly utilizing cuML and CUDA based libraries for GPU manipulation of data such as CuPy and cudf. I will be analyzing a dataset that shows crop planting recommendations. This project is a multi-class classification problem that uses classifiers built into cuML. Currently, those classifers are: Random Forest, Logistic Regression, LinearSVC, and K Nearest Neighbors.

### Dependencies
This requires the user to have a CUDA capable graphics card and have CUDA installed on their machine. This also requires the cuDF and cuML libraries to perform the data manipulation and analysis via machine learning algorithms.

### How to run
There are two scripts included here. Simply navigate to the directory and run the script "run.sh" in the terminal. This will run the program WITHOUT cross validation using the hyper-parameter values that I found performing cross validation myself. However, if you wish to run CV for yourself, run the script "run_cv.sh" in the terminal. Cross validation takes some time so be prepared.

### Known Issues
1. When performing cross validation and with the classifiers for Linear Regression and Support Vector Classification several warnings pop up. These warning are mostly about iterations and poor hyper-parameter selection. 

2. I am having difficulty with the dask implementation of K Nearest Neighbors. It never finishes training and instead restarts the program. I need to debug this further, so as of right now it is commented out.