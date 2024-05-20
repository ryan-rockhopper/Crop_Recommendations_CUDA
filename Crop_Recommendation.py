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
import DataPreprocessor.Preprocessor as pp

from cuml.metrics import accuracy_score
from cuml.metrics import confusion_matrix
from cuml.ensemble import RandomForestClassifier
from cuml.linear_model import LogisticRegression
from OtherMetrics import findF1

#Set random seeds
cp.random.seed(31)
torch.manual_seed(31)

# Set random seed for GPU if available
if torch.cuda.is_available():
    torch.cuda.manual_seed(31)

fullData = cd.read_csv('./data/Crop_Recommendation.csv')
pp.encodeLabels(fullData, 'Crop')
trainingData, testData = pp.createTrainingAndTestSets(fullData, 0.2)

print(f'Full data rows      = {len(fullData)}')
print(f'Training data rows  = {len(trainingData)}')
print(f'Test data rows      = {len(testData)}')

trainingFeatures, trainingLabels = pp.extractLabels(trainingData, 'Crop')
testFeatures, testLabels         = pp.extractLabels(testData, 'Crop')

#DEBUG: Trying to get 1 dimension for labels
#label0 = testLabels.to_cupy()
#label0 = label0[1]
label0 = trainingLabels.values.flatten()

#TODO Cross validation, Accuracy is low, look into it.
#TODO: Implement linear regression, decision tree, random forest, gradient boosting machine, support vector regression, and neural network

#~~RANDOM FOREST~~
print("\n\nTraining Decision Tree Regressor model")
model   = RandomForestClassifier(max_depth=5, n_estimators=50, random_state=31, n_streams=1, split_criterion=1) #TODO: for CV, check gini index (split_criterion=0) and entropy (1)
start   = time.time()
model.fit(trainingFeatures, trainingLabels.values.flatten())
end     = time.time()
elapsed = end-start
print(f"Training the Random Forest took {round(elapsed, 2)} seconds.")

print("Making predictions on the testing set")
start   = time.time()
predictions = model.predict(testFeatures)
end     = time.time()
elapsed = end-start
print(f"Making predictions took {round(elapsed,2)} seconds.")

#Metrics
accuracy = accuracy_score(testLabels, predictions)
#confusion_matrix requires int32 or int64, as well as 1D arrays, hence the cast and flattening.
cfMatrix = confusion_matrix(testLabels.astype('int32').values.flatten(), predictions.astype('int32').values.flatten())
averageF1 = findF1(cfMatrix)
print(f"Accuracy for the Random Forest is:          {round(accuracy, 4)}")
print(f"Average F1 score for the Random Forest is:  {round(averageF1, 4)}")



#~~LOGISTIC REGRESSION~~
print("\n\nTraining Logistic Regression model")
model   = LogisticRegression()
start   = time.time()
model.fit(trainingFeatures, trainingLabels.values.flatten())
end     = time.time()
elapsed = end-start
print(f"Training Logistic Regression took {round(elapsed, 2)} seconds.")

print("Making predictions on the testing set")
start   = time.time()
predictions = model.predict(testFeatures)
end     = time.time()
elapsed = end-start
print(f"Making predictions took {round(elapsed,2)} seconds.")

#Metrics
accuracy = accuracy_score(testLabels, predictions)
cfMatrix = confusion_matrix(testLabels.astype('int32').values.flatten(), predictions.astype('int32').values.flatten())
averageF1 = findF1(cfMatrix)
print(f"Accuracy for Logistic Regression is:            {round(accuracy, 4)}")
print(f"Average F1 score for Logistic Regression is:    {round(averageF1, 4)}")