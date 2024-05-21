'''
Author:     Ryan Dalrymple
Date:       05/21/24
Summary:    Contains methods that will perform cross validation on the various classifiers used in Crop_Recommendation.py
            in order to find the best hyperparameters. 
'''

import DataPreprocessor.Preprocessor as pp

from cuml.metrics       import confusion_matrix
from cuml.ensemble      import RandomForestClassifier
from cuml.linear_model  import LogisticRegression
from cuml.svm           import LinearSVC
from cuml.neighbors     import KNeighborsClassifier
from OtherMetrics       import findF1