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
        highestF1: The highest F1 score when the optimal hyperparameters are used
    """
    bestDepth       = -1
    bestEstimators  = -1
    bestCriteron    = -1
    highestF1       = -1

    dataVector = pp.splitData(trainingData, folds)

    for i in range (len(depths)):
        print(f'Current depth: {depths[i]}')
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
                    
        print(f'{round((((i+1) / len(depths)) * 100), 2)}% complete with CV.')
    return bestDepth, bestEstimators, bestCriteron, highestF1

def logisticRegression_CV(trainingData, tols, reg_C, iterLimits, folds, labelCol):
    """Performs cross validation for cuML's logistic regression model.

    Args:
        trainingData (cuDF dataframe): The training data which will be split and have cross validation performed on it.
        tols (list[float]): Contains different tolerances for an exit condition on the logistic regression training.
        reg_C (list[float]): Contains different values for C (the regularization constant)
        iterLimits (list[int]): Contains different values for the maximum amount of iterations permitted during logistic regression training
                                if the tolerance is never met
        folds (int): The number of folds to split the training data into for k-fold cross validation.
        labelCol (string): The header of the column which contains the labels in trainingData.

    Returns:
        bestTol (float): The tolerance associated with the highest F1 score.
        bestC (float): The regularization constant associated with the highest F1 score.
        bestIterations: The maximum iteration limit associated with the highest F1 score.
        highestF1: The highest F1 score when the optimal hyperparameters are used
    """
    bestTol         = -1
    bestC           = -1
    bestIterations  = -1
    highestF1       = -1

    dataVector = pp.splitData(trainingData, folds)

    for i in range (len(tols)):
        print(f'Current tolerance: {tols[i]}')
        for j in range (len(reg_C)):
            for l in range (len(iterLimits)):
                avgF1 = 0

                for k in range (len(dataVector)):
                    #Concatenate all but the kth dataframe for cross validation
                    trainingDataFrames = [df for index, df in enumerate(dataVector) if index != k]
                    trainingDataFrames = cd.concat(trainingDataFrames)

                    trainingFeatures, trainingLabels = pp.extractLabels(trainingDataFrames, labelCol)
                    testFeatures, testLabels         = pp.extractLabels(dataVector[k], labelCol)

                    model = LogisticRegression(tol=tols[i], C=reg_C[j], max_iter=iterLimits[l], verbose=False)
                    model.fit(trainingFeatures, trainingLabels.values.flatten())
                    predictions = model.predict(testFeatures)

                    cfMatrix = confusion_matrix(testLabels.astype('int32').values.flatten(), predictions.astype('int32').values.flatten())
                    avgF1   += findF1(cfMatrix)

                avgF1 /= len(dataVector)
                if avgF1 > highestF1:
                    bestTol         = tols[i]
                    bestC           = reg_C[j]
                    bestIterations  = iterLimits[l]
                    highestF1       = avgF1
                    
        print(f'{round((((i+1) / len(tols)) * 100), 2)}% complete with CV.')
    return bestTol, bestC, bestIterations, highestF1

def linearSVC_CV(trainingData, penalties, losses, intrPenalize, tols, reg_C, iterLimits, folds, labelCol):
    """Performs cross validation for the Linear SVC classifier.

    Args:
        trainingData (cuDF dataframe): The training data which will be split and have cross validation performed on it.
        penalties (list[string]): Contains either l1 or l2 which are the two regularization terms of the target function.
        losses (list[string]): Contains either squared_hinge or hinge which are the two loss functions that are applicable.
        intrPenalize (list[bool]): Contains either True or False which determines if the bias term is penalized in the same way by the regularization term.
        tols (list[float]): Contains different tolerances for an exit condition on the logistic regression training.
        reg_C (list[float]): Contains different values for C (the regularization constant)
        iterLimits (list[int]): Contains different values for the maximum amount of iterations permitted during logistic regression training
                                if the tolerance is never met
        folds (int): The number of folds to split the training data into for k-fold cross validation.
        labelCol (string): The header of the column which contains the labels in trainingData.

    Returns:
        bestPenalty (string): The penalty associated with the highest F1 score.
        bestLoss (string): The loss function associated with the highest F1 score.
        penalize_intercept: True if that is associated with the highest F1 score, false otherwise.
        bestTol (float): The tolerance associated with the highest F1 score.
        bestC (float): The regularization constant associated with the highest F1 score.
        bestIterations: The maximum iteration limit associated with the highest F1 score.
        highestF1: The highest F1 score when the optimal hyperparameters are used
    """
    bestPenalty         = -1
    bestLoss            = -1
    penalize_intercept  = False
    bestTol             = -1
    bestC               = -1
    bestIterations      = -1
    highestF1           = -1

    dataVector = pp.splitData(trainingData, folds)

    for t in range (len(penalties)):
        for y in range (len(losses)):
            for u in range (len(intrPenalize)):
                for i in range (len(tols)):
                    for j in range (len(reg_C)):
                        for l in range (len(iterLimits)):
                            avgF1 = 0

                            for k in range (len(dataVector)):
                                #Concatenate all but the kth dataframe for cross validation
                                trainingDataFrames = [df for index, df in enumerate(dataVector) if index != k]
                                trainingDataFrames = cd.concat(trainingDataFrames)

                                trainingFeatures, trainingLabels = pp.extractLabels(trainingDataFrames, labelCol)
                                testFeatures, testLabels         = pp.extractLabels(dataVector[k], labelCol)

                                model = LinearSVC(penalty=penalties[t], loss=losses[y], penalized_intercept=intrPenalize[u], tol=tols[i], C=reg_C[j], max_iter=iterLimits[l])
                                model.fit(trainingFeatures.values, trainingLabels.values.flatten())
                                predictions = model.predict(testFeatures)

                                cfMatrix = confusion_matrix(testLabels.astype('int32').values.flatten(), predictions.astype('int32').values.flatten())
                                avgF1   += findF1(cfMatrix)

                            avgF1 /= len(dataVector)
                            if avgF1 > highestF1:
                                bestPenalty         = penalties[t]
                                bestLoss            = losses[y]
                                penalize_intercept  = intrPenalize[u]
                                bestTol             = tols[i]
                                bestC               = reg_C[j]
                                bestIterations      = iterLimits[l]
                                highestF1           = avgF1
                    
    return bestPenalty, bestLoss, penalize_intercept, bestTol, bestC, bestIterations, highestF1

def KNN_CV(trainingData, neighborCounts, metrics, folds, labelCol):
    """Performs Cross Validation for the K Nearest Neighbors classifier.

    Args:
        trainingData (cuDF dataframe): The training data which will be split and have cross validation performed on it.
        neighborCounts (list[int]): A list containing integers that are the amount of neighbors (K) for the classifier to consider
        metrics (list[string]): A list containing strings which represent different different ways to measure distance to the neighbors
        folds (int): The number of folds to split the training data into for k-fold cross validation.
        labelCol (string): The header of the column which contains the labels in trainingData.

    Returns:
        bestK: The amount of neighbors associated with the highest F1 score
        bestMetric: The measurement that results in the highest F1 score
        highestF1: The highest F1 score when the optimal hyperparameters are used
    """
    bestK       = -1
    bestMetric  = -1
    highestF1   = -1

    dataVector = pp.splitData(trainingData, folds)

    for i in range (len(neighborCounts)):
        print(f'Current K value: {neighborCounts[i]}')
        for j in range (len(metrics)):
            avgF1 = 0

            for k in range (len(dataVector)):
                #Concatenate all but the kth dataframe for cross validation
                trainingDataFrames = [df for index, df in enumerate(dataVector) if index != k]
                trainingDataFrames = cd.concat(trainingDataFrames)

                trainingFeatures, trainingLabels = pp.extractLabels(trainingDataFrames, labelCol)
                testFeatures, testLabels         = pp.extractLabels(dataVector[k], labelCol)

                model   = KNeighborsClassifier(n_neighbors=neighborCounts[i], metric=metrics[j])
                model.fit(trainingFeatures.values, trainingLabels.values.flatten())
                predictions = model.predict(testFeatures)

                cfMatrix = confusion_matrix(testLabels.astype('int32').values.flatten(), predictions.astype('int32').values.flatten())
                avgF1   += findF1(cfMatrix)

            avgF1 /= len(dataVector)
            if avgF1 > highestF1:
                bestK       = neighborCounts[i]
                bestMetric  = metrics[j]
                highestF1   = avgF1
                    
        print(f'{round((((i+1) / len(neighborCounts)) * 100), 2)}% complete with CV.')
    return bestK, bestMetric, highestF1
