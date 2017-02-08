# -*- coding: utf-8 -*-
# Mini Project 2
# Text Classification using Random Forests Classifier of Scikit Learn
# Train Test Split Ratio 80:20, Trees in Random Forest = 100
import pandas as pd
import numpy
from nltk.corpus import stopwords
import re
from bs4 import BeautifulSoup

train = pd.read_csv("train_input.csv", header=0)

def conversation_words( raw_text ):
    # Function to pre process text data by removing html tags, punctuation 
    # and stop word
    #
    # Removing HTML tags
    text = BeautifulSoup(raw_text).get_text() 
    #
    # Removing punctuation and non letter characters        
    filter_text = re.sub("[^a-zA-Z]", " ", text) 
    #
    # Converting to lower case and splitting into individual words
    words = filter_text.lower().split()  
    #
    #Removing stop words                       
    stops = set(stopwords.words("english"))                  
    #
    proper_words = [w for w in words if not w in stops]   
    #
    # 
    return( " ".join( proper_words )) 
    
clean_text = conversation_words( train["conversation"][0] )
print clean_text

total_conv = train["conversation"].size
clean_train_conv = []
for i in xrange( 0, total_conv ):
    clean_train_conv.append( conversation_words( train["conversation"][i] ) )

print "Creating the bag of words...\n"
from sklearn.feature_extraction.text import CountVectorizer

# Initializing the "CountVectorizer" object 
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000) 

train_features = vectorizer.fit_transform(clean_train_conv)
train_features = train_features.toarray()
print train_features.shape

train_y = pd.read_csv("train_output.csv", header=0)
train_y.columns.values

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split( train_features, train_y, test_size=0.2, random_state=0 ) 


from sklearn.ensemble import RandomForestClassifier

 # Initializing a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators = 100) 

forest = forest.fit( X_train, y_train["category"])

# Validaiton on X_test
predict_validation=forest.predict(X_test)
from sklearn import metrics
print(metrics.classification_report(y_test["category"], predict_validation))

# Final Testing on unseen data
# Reading the final test data
test = pd.read_csv("test_input.csv", header=0)
print test.shape

# Creating an empty list and appending the clean conversations one by one
count_conv = len(test["conversation"])
clean_test_conv = [] 

print "Cleaning and parsing the conversations...\n"
for i in xrange(0,count_conv):
    if( (i+1) % 1000 == 0 ):
        print "Conversation %d of %d\n" % (i+1, count_conv)
    clean_convt = conversation_words( test["conversation"][i] )
    clean_test_conv.append( clean_convt )

# Get a bag of words for the test set, and convert to a numpy array
test_features = vectorizer.transform(clean_test_conv)
test_features = test_features.toarray()

# Use the random forest for making category predictions
result = forest.predict(test_features)

# Copying the results to a pandas dataframe with an "id" column and
# a "category" column
output = pd.DataFrame( data={"id":test["id"], "category":result} )
#
cols=output.columns.tolist()
cols=cols[-1:]+cols[:-1]
output=output[cols]

# Use pandas to write the comma-separated output file
output.to_csv( "Bag_of_Words_Forest_100_tress.csv", index=False)
