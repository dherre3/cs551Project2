# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
import re
from bs4 import BeautifulSoup

train = pd.read_csv("trainining_dataset.csv", header=0)
ps = LancasterStemmer()
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
    #filter all the words by the stopwords function and additionally stem it
    proper_words = [ps.stem(w) for w in words if not w in stops]   
    # 
    return( " ".join( proper_words )) 
    
clean_text = conversation_words( train["conversation"][0] )
print (clean_text)

total_conv = train["conversation"].size
clean_train_conv = []
for i in range(total_conv):
   train["conversation"][i] = ( conversation_words( train["conversation"][i] ) )

train.to_csv('Input_train.csv',header=True,encoding='utf-8')



##print ("Creating the bag of words...\n")
##from sklearn.feature_extraction.text import CountVectorizer

### Initializing the "CountVectorizer" object 
##vectorizer = CountVectorizer(analyzer = "word",   \
##                             tokenizer = None,    \
##                             preprocessor = None, \
##                             stop_words = None,   \
##                             max_features = 5000) 

##train_features = vectorizer.fit_transform(clean_train_conv)
##train_features = train_features.toarray()
##print (train_features.shape)




