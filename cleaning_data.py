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
#==============================================================================
# Script Parameters
#==============================================================================
trainFilepath = './data/Input_train_pre_processed_stem.csv'
#targetFilepath = './data/train_output.csv'
C = 1.0 # SVM regularization parameter
#==============================================================================
# Importing Dataset
#==============================================================================
datasetTrainInput = pd.read_csv(trainFilepath)
#datasetTrainOutput = datasetTrainInput['category'].values
datasetTestInput = pd.read_csv('./data/Test_set_pre_processed_stem.csv')

conversationsTrain = datasetTrainInput['conversation'].values
#conversationsTrain = np.array([cl.cleanData(x) for x in conversationsTrain]) 

conversationsTest = datasetTestInput['conversation'].values
#conversationsTest = np.array([cl.cleanData(x) for x in conversationsTest]) 

train_output = datasetTrainInput['category'].values
train_input = conversationsTrain
test_input = conversationsTest



#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(train_input, train_output, test_size=0.2, random_state=0 ) 


from sklearn.feature_extraction.text import CountVectorizer
#count_vect = CountVectorizer()
#X_train_counts = count_vect.fit_transform(X_train)
#print(X_train_counts)


from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn import tree
text_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),
                      ('clf', MultinomialNB()) ])
text_clf = text_clf.fit(train_input, train_output)
predicted = text_clf.predict(test_input)
print(predicted)
ids = datasetTestInput['id'].values
print(ids.shape, predicted.shape)
prediction = np.array([ids,predicted]).T
prediction_dataframe = pd.DataFrame(prediction, columns=['id','category'])          
prediction_dataframe.to_csv('./data/test_prediction_v1_multinomial_92.csv', index=False)                      

from sklearn import metrics
predictedTrain = text_clf.predict(train_input)
print(metrics.classification_report(train_output, predictedTrain))