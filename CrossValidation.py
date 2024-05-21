'''
Author:     Ryan Dalrymple
Date:       05/21/24
Summary:    Contains methods that will perform cross validation on the various classifiers used in Crop_Recommendation.py
            in order to find the best hyperparameters. 
'''

import DataPreprocessor.Preprocessor as pp
import cudf as cd

from cuml.metrics       import confusion_matrix
from cuml.ensemble      import RandomForestClassifier
from cuml.linear_model  import LogisticRegression
from cuml.svm           import LinearSVC
from cuml.neighbors     import KNeighborsClassifier
from OtherMetrics       import findF1

def randomForest_CV(trainingData, depths, estimators, split_criterions, folds, labelCol):
    """Performs cross validation for cuML's RandomForest classifier.

    Args:
        trainingData (cuDF dataframe): The training data which will be split and have cross validation performed on it.
        depths (list[int]): A list of the depths to check
        estimators (list[int]): A list of the n_estimators to check
        split_criterions (list[int]): A list containing the ints 0 and 1. 0 performs tree splits using gini index while 1 uses entropy as splitting criteron
        folds (int): The number of folds to split the training data into for k-fold cross validation.
        labelCol (string): The header of the column which contains the labels in trainingData.

    Returns:
        bestDepth (int): The depth associated with the highest F1 score.
        bestEstimators (int): The n_estimators associated with the highest F1 score.
        bestCriteron (int): A 0 or 1 depending on which is associated with the highest F1 score.
    """
    bestDepth       = -1
    bestEstimators  = -1
    bestCriteron    = -1
    highestF1       = -1

    dataVector = pp.splitData(trainingData, folds)

    for i in range (len(depths)):
        for j in range (len(estimators)):
            for l in range (len(split_criterions)):
                avgF1 = 0

                for k in range (len(dataVector)):
                    #Concatenate all but the kth dataframe for cross validation
                    trainingDataFrames = [df for index, df in enumerate(dataVector) if index != k]
                    trainingDataFrames = cd.concat(trainingDataFrames)

                    trainingFeatures, trainingLabels = pp.extractLabels(trainingDataFrames, labelCol)
                    testFeatures, testLabels         = pp.extractLabels(dataVector[k], labelCol)

                    model = RandomForestClassifier(max_depth=depths[i], n_estimators=estimators[j], random_state=31, n_streams=1, split_criterion=split_criterions[l])
                    model.fit(trainingFeatures, trainingLabels.values.flatten())
                    predictions = model.predict(testFeatures)

                    cfMatrix = confusion_matrix(testLabels.astype('int32').values.flatten(), predictions.astype('int32').values.flatten())
                    avgF1   += findF1(cfMatrix)

                avgF1 /= len(dataVector)
                if avgF1 > highestF1:
                    bestDepth       = depths[i]
                    bestEstimators  = estimators[j]
                    bestCriteron    = split_criterions[l]
                    highestF1       = avgF1
                    
    return bestDepth, bestEstimators, bestCriteron, highestF1