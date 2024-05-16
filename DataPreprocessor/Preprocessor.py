'''
Author: Ryan Dalrymple

This is used to split a pandas dataframe into smaller, evenly sized dataframes
for use in cross validation.
'''
import sys
import cudf as cd

def splitData(fullData, numberOfSplits):
    #Shuffles the data in the dataframe and resets the index so it starts at 0 again.
    shuffledData    = fullData.sample(frac=1, random_state=31).reset_index(drop=True)
    rowCount        = shuffledData.shape[0]
    splitSize       = rowCount // numberOfSplits

    #Splits into numberOfSplits dataframes from the shuffled data
    segments = [shuffledData.iloc[i*splitSize:(i+1)*splitSize] for i in range(numberOfSplits)]

    return segments

def createTrainingAndTestSets(dataset, testPercentage):
    """
    Given a dataset, this function will split it into two parts, so one can be reserved for a test set.

    Args:
        dataset (cudf dataframe): The full dataset
        testPercentage (float): The percentage of the dataset that should be reserved for a test set, for example 0.2 (20%)

    Returns:
        The two datasets of trainingData and testData
    """
    #Shuffle original data
    dataset = dataset.sample(frac=1, random_state=31).reset_index(drop=True)

    testData = dataset.sample(frac=testPercentage, random_state=31)
    testData = testData.reset_index(drop=True)

    trainingData = dataset.drop(testData.index)
    trainingData = trainingData.reset_index(drop=True)

    return trainingData, testData

def extractLabels(dataset, labelColumn):
    """Extracts the labels from a dataset so models can be trained and tested on them

    Args:
        dataset (cudf dataframe): The dataframe from which the labels are to be extracted
        labelColumn (string): The header of the column which contains the labels

    Returns:
        The original dataset with the label column removed and the label column as a separate dataset set 'y'
    """
    if labelColumn in dataset.columns:
        y = dataset[[labelColumn]].copy()
        dataset.drop(columns=[labelColumn], inplace=True)
        return dataset, y
    
    else:
        print(f"The dataset did not contain a column with the header {labelColumn}")