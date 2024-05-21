'''
Author:     Ryan Dalrymple
Date:       May 16 2024
Overview:   This is the 'main' file for the Crop_Recommendation analysis project. All algorithms will be run in this file
'''

import time
import cuml
import cupy as cp
import cudf as cd
import CrossValidation as cv
import DataPreprocessor.Preprocessor as pp

from cuml.metrics       import accuracy_score
from cuml.metrics       import confusion_matrix
from cuml.ensemble      import RandomForestClassifier
from cuml.linear_model  import LogisticRegression
from cuml.svm           import LinearSVC
from cuml.neighbors     import KNeighborsClassifier
from OtherMetrics       import findF1


import dask_cuda
from dask.distributed       import Client
from cuml.dask.neighbors    import KNeighborsClassifier as KNN
from cuml.dask.common       import to_dask_cudf


#Set random seeds
cp.random.seed(31)
performCrossValidation = True

fullData = cd.read_csv('./data/Crop_Recommendation.csv')
pp.encodeLabels(fullData, 'Crop')
trainingData, testData = pp.createTrainingAndTestSets(fullData, 0.2)

print(f'Full data rows      = {len(fullData)}')
print(f'Training data rows  = {len(trainingData)}')
print(f'Test data rows      = {len(testData)}')

trainingFeatures, trainingLabels = pp.extractLabels(trainingData, 'Crop')
testFeatures, testLabels         = pp.extractLabels(testData, 'Crop')



#~~RANDOM FOREST~~
print("\n\nTraining Random Forest model")

#Found from Cross Validation
bestDepth       = 5
bestEstimators  = 50
bestCriteron    = 1
cvF1            = -1

if performCrossValidation:
        depths      = [2, 5, 10, 15, 25]
        estimators  = [5, 10, 25, 50, 100]
        criterions   = [0, 1]

        print(f'Beginning cross validation for Random Forest.')
        start   = time.time()
        bestDepth, bestEstimators, bestCriteron, cvF1 = cv.randomForest_CV(trainingData, depths, estimators, criterions, 10, 'Crop')
        end     = time.time()
        elapsed = end-start
        print(f'Performing cross validation for Random Forst took {round(elapsed, 2)} seconds.')
        print(f'The results from CV are printed below:')
        print(f'Best Depth:     {bestDepth}')
        print(f'Best number of Estimators:      {bestEstimators}')
        print(f'Best split criterion: Gini Index' if bestCriteron == 0 else 'Best split criterion: Entropy')
        print(f'Associated F1 score: {cvF1}')

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
accuracy  = accuracy_score(testLabels, predictions)
#confusion_matrix requires int32 or int64, as well as 1D arrays, hence the cast and flattening.
cfMatrix  = confusion_matrix(testLabels.astype('int32').values.flatten(), predictions.astype('int32').values.flatten())
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
accuracy  = accuracy_score(testLabels, predictions)
cfMatrix  = confusion_matrix(testLabels.astype('int32').values.flatten(), predictions.astype('int32').values.flatten())
averageF1 = findF1(cfMatrix)
print(f"Accuracy for Logistic Regression is:            {round(accuracy, 4)}")
print(f"Average F1 score for Logistic Regression is:    {round(averageF1, 4)}")



#~~SUPPORT VECTOR CLASSIFICATION~~
print("\n\nTraining Support Vector Classification (SVC) model")
model   = LinearSVC() #TODO: Check multi_class='ovo' when implemented (Not possible as of 5/20/24), check hinged loss vs squared hinge loss in CV
start   = time.time()
model.fit(trainingFeatures.values, trainingLabels.values.flatten()) #SVC expects array not dataframe, hence trainingFeatures.values
end     = time.time()
elapsed = end-start
print(f"Training SVC took {round(elapsed, 2)} seconds.")

print("Making predictions on the testing set")
start   = time.time()
predictions = model.predict(testFeatures)
end     = time.time()
elapsed = end-start
print(f"Making predictions took {round(elapsed,2)} seconds.")

#Metrics
accuracy  = accuracy_score(testLabels, predictions)
cfMatrix  = confusion_matrix(testLabels.astype('int32').values.flatten(), predictions.astype('int32').values.flatten())
averageF1 = findF1(cfMatrix)
print(f"Accuracy for SVC is:            {round(accuracy, 4)}")
print(f"Average F1 score for SVC is:    {round(averageF1, 4)}")



#~~K-NEAREST NEIGHBORS~~
print("\n\nTraining K-Nearest Neighbors (KNN) model")
model   = KNeighborsClassifier(n_neighbors=5) #TODO: n_neighbors is hyperparameter
start   = time.time()
model.fit(trainingFeatures.values, trainingLabels.values.flatten()) #SVC expects array not dataframe, hence trainingFeatures.values
end     = time.time()
elapsed = end-start
print(f"Training KNN took {round(elapsed, 2)} seconds.")

print("Making predictions on the testing set")
start   = time.time()
predictions = model.predict(testFeatures)
end     = time.time()
elapsed = end-start
print(f"Making predictions took {round(elapsed,2)} seconds.")

#Metrics
accuracy  = accuracy_score(testLabels, predictions)
cfMatrix  = confusion_matrix(testLabels.astype('int32').values.flatten(), predictions.astype('int32').values.flatten())
averageF1 = findF1(cfMatrix)
print(f"Accuracy for KNN is:            {round(accuracy, 4)}")
print(f"Average F1 score for KNN is:    {round(averageF1, 4)}")

'''
print("\n\nTraining Dask K-Nearest Neighbors (KNN) model")

"""
Dask: Dask is a flexible parallel computing library for Python that enables scalable, distributed computing. 
        It provides dynamic task scheduling and parallel collections (such as arrays, dataframes, and lists) 
        that extend beyond the memory and processing limitations of a single machine.

CUDA: CUDA is a parallel computing platform and programming model developed by NVIDIA for accelerating computations on NVIDIA GPUs. 
        It allows developers to harness the power of GPUs for general-purpose computing tasks.

Combining Dask with CUDA creates a distributed computing environment that can efficiently utilize GPUs for parallel processing.
"""
#Need a Dask CUDA cluster for this classifier.
cluster = dask_cuda.LocalCUDACluster()
client  = Client(cluster)

trainingDask        = to_dask_cudf(trainingFeatures)
trainingLabelsDask  = to_dask_cudf(trainingLabels)
testDask            = to_dask_cudf(testFeatures)
testLabelsDask      = to_dask_cudf(testLabels)

model   = KNN(n_neighbors=5) #TODO: n_neighbors is hyperparameter
start   = time.time()
model.fit(trainingDask, trainingLabelsDask)
end     = time.time()
elapsed = end-start
print(f"Training Dask KNN took {round(elapsed, 2)} seconds.")

print("Making predictions on the testing set")
start   = time.time()
predictions = model.predict(testDask)
end     = time.time()
elapsed = end-start
print(f"Making predictions took {round(elapsed,2)} seconds.")

#Metrics
#accuracy  = accuracy_score(testLabels, predictions)
accuracy = (predictions == testLabelsDask).mean().compute()
#cfMatrix  = confusion_matrix(testLabels.astype('int32').values.flatten(), predictions.astype('int32').values.flatten())
#averageF1 = findF1(cfMatrix)
print(f"Accuracy for Dask KNN is:            {round(accuracy, 4)}")
#print(f"Average F1 score for KNN is:    {round(averageF1, 4)}")
client.close()
cluster.close() #TODO: Dask KNN does not finish training and restarts the entire program. Not sure what is going on.
'''