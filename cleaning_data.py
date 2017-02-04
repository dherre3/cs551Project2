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
trainFilepath = './data/train_input.csv'
targetFilepath = './data/train_output.csv'
C = 1.0 # SVM regularization parameter
#==============================================================================
# Importing Dataset
#==============================================================================
datasetTrainInput = pd.read_csv(trainFilepath)
datasetTrainOutput = pd.read_csv(targetFilepath)

conversations = datasetTrainInput['conversation'].values
conversations = np.array([cl.cleanData(x) for x in conversations]) 
train_output = datasetTrainOutput['category'].values
train_input = conversations


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_input, train_output, test_size=0.2, random_state=0 ) 


#from sklearn.feature_extraction.text import CountVectorizer
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
text_clf = text_clf.fit(X_train, y_train)
predicted = text_clf.predict(X_test)
print(np.mean(predicted == y_test))
from sklearn import metrics
print(metrics.classification_report(y_test, predicted))