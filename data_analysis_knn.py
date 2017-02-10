"""
Created on Sat Feb  4 13:31:39 2017

@author: davidherrera
"""
#==============================================================================
# Importing Libraries 
#==============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cleaning_library as cl
import nltk
import kNN as knn
#==============================================================================
# Script Parameters
#==============================================================================
#trainFilepath = './data/Input_train_pre_processed_stem.csv'
trainFilepath = './data/train_input.csv'

targetFilepath = './data/train_output.csv'

targetTestFilepath = './data/test_input.csv'
C = 1.0 # SVM regularization parameter
#==============================================================================
# Importing Dataset
#==============================================================================
datasetTrainInput = pd.read_csv(trainFilepath)
datasetTrainOutput = pd.read_csv(targetFilepath)
#datasetTestInput = pd.read_csv('./data/Test_set_pre_processed_stem.csv')
datasetTestInput = pd.read_csv(targetTestFilepath)

conversationsTrain = datasetTrainInput['conversation'].values
conversationsTrain = np.array([cl.cleanData(x) for x in conversationsTrain]) 

conversationsTest = datasetTestInput['conversation'].values
conversationsTest = np.array([cl.cleanData(x) for x in conversationsTest]) 

train_output = datasetTrainOutput['category'].values
train_input = conversationsTrain
test_input = conversationsTest

#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(train_input, train_output, test_size=0.2, random_state=0 ) 


from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer(max_features=1000)
X_train_counts = count_vect.fit_transform(train_input)
X_test_counts = count_vect.transform(test_input)

from sklearn.feature_extraction.text import TfidfTransformer
Tfid_vect =TfidfTransformer()
tfid_result = Tfid_vect.fit_transform(X_train_counts)
X_test_counts = Tfid_vect.transform(X_test_counts)

x_train = X_train_counts.toarray();
x_test = X_test_counts.toarray();
prediction = knn.kNN(x_train,train_output, x_test,10)
ids = datasetTestInput['id'].values
print(ids.shape, predicted.shape)
prediction = np.array([ids,predicted]).T
prediction_dataframe = pd.DataFrame(prediction, columns=['id','category'])          
prediction_dataframe.to_csv('./data/test_prediction_v1_knn.csv', index=False)