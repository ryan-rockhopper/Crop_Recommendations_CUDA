'''
Author:     Ryan Dalrymple
Date:       May 16 2024
Overview:   This is the 'main' file for the IMDB analysis project. All algorithms will be run in this file
'''

import time
import torch
import cupy as cp
import cudf as cd
import cuml 

from DataPreprocessor.Preprocessor import splitData
from DataPreprocessor.Preprocessor import createTrainingAndTestSets
from DataPreprocessor.Preprocessor import extractLabels
from cuml.metrics import mean_squared_error

#Set random seeds
cp.random.seed(31)
torch.manual_seed(31)

# Set random seed for GPU if available
if torch.cuda.is_available():
    torch.cuda.manual_seed(31)

fullData = cd.read_csv('./data/movie_dataset.csv')
fullData.drop(columns=['overview'], inplace=True) #Drop the movie description from the dataframe, not a useful feature.
trainingData, testData = createTrainingAndTestSets(fullData, 0.2)

print(f'Full data rows      = {len(fullData)}')
print(f'Training data rows  = {len(trainingData)}')
print(f'Test data rows      = {len(testData)}')

trainingFeatures, trainingLabels = extractLabels(trainingData, 'vote_average')
testFeatures, testLabels         = extractLabels(testData, 'vote_average')

#TODO: Decision trees not supported. Need to transform data into numerical values before using cuML algorithms.
#~~DECISION TREE~~ TODO Cross validation
print("Training Decision Tree Regressor model")
model   = cuml.DecisionTreeRegressor(max_depth=5)
start   = time.time()
model.fit(trainingFeatures, trainingLabels)
end     = time.time()
elapsed = end-start
print(f"Training the Decision Tree took {round(elapsed, 2)} seconds.")

print("Making predictions on the testing set")
start   = time.time()
predictions = model.predict(testFeatures)
end     = time.time()
elapsed = end-start
print(f"Making predictions took {round(elapsed,2)} seconds.")

mse = mean_squared_error(testLabels, predictions)
print(f"Mean Squared Error for the Decision Tree is: {round(mse, 4)}")

#TODO: Implement linear regression, decision tree, random forest, gradient boosting machine, support vector regression, and neural network
