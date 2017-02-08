# -*- coding: utf-8 -*-
import re, pickle
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk import FreqDist
from nltk import word_tokenize
from nltk.stem.lancaster import LancasterStemmer
from bs4 import BeautifulSoup
from collections import OrderedDict
from math import log
from sklearn.model_selection import train_test_split




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
    proper_words = [ps.stem(w) for w in words if not w in stops]   
    # 
    return( " ".join( proper_words )) 
    

def pre_process_words (dataset, filename):
    print()
    print('Preprocessing the file')
    total_conv = dataset["conversation"].size
    for i in range(total_conv):
        dataset["conversation"][i] = ( conversation_words( dataset["conversation"][i] ) )

    ##outputs the preprocessed file
    dataset.to_csv(filename, header=True,encoding='utf-8')
    return dataset

##------------------------------------------------------------------
#          A function that calculates and returns:
# "bag_of_words" a dictionary that returns the number of occurrences of word i given category c;
# "unique words" the value of how many words we are considering (n * # categories)
# "number_per_feature" - dictionary with the number of words given a category
##-----------------------------------------------------------------
def get_bag_words(dataset, n):

    print('Obtaining most common words')
    #list of labels

    labels = np.unique (dataset['category']) 
    
    bag_of_words = {}
    unique_words = 0
    number_per_feature = {}
    for category in labels:
        bag_of_words [category] = {}
        #conversation_per_feature= []
        auxiliar = dataset [dataset ['category']==category] ['conversation']
        conversation_per_feature = [conversation for conversation in auxiliar]
        conversation_per_feature = word_tokenize( ''.join(conversation_per_feature) )
        most_common_per_feature =  FreqDist (conversation_per_feature)
        
        #unique_words += len(most_common_per_feature.most_common(50000))
        unique_words += n
        #unique_words += len(np.unique(conversation_per_feature))

        total_number_words = 0
        for item in most_common_per_feature.most_common(n):
            total_number_words += item[1]

        number_per_feature[category] = total_number_words
      
        bag_of_words [category] = most_common_per_feature
    
    #returns a dictionary of tuples   
    return bag_of_words , unique_words , number_per_feature 

##------------------------------------------------------------------
# Calculate the probability of a certain category in the training set
##------------------------------------------------------------------
def category_prob (train_dataset, category):
    prob_category = len(train_dataset [ train_dataset['category']==category] ['category'])/len(train_dataset.index)
    return prob_category

##------------------------------------------------------------------
# Calculates the log(P(category))  - returns a dictionary in which the category is the key
##------------------------------------------------------------------
def log_prob_category (train_dataset):
    category_labels = np.unique(train_dataset['category'])

    cat_prob = {}
    for category in category_labels:
        cat_prob [category] = log (category_prob (train_dataset, category))

    return cat_prob

##------------------------------------------------------------------
#          Makes the prediction, given the trained parameters:
# "bag_of_words" (returns the number of occurrences of word i given category c);
# "cat_prob" a dictionary that returns the probability of each category
# "total_number_words" - total number of words in the whole training set
# "number_per_feature" - dictionary with the number of words given a category
# "filename" - path to save the file with the predictions
##------------------------------------------------------------------

def prediction (test_dataset, bag_of_words, cat_prob , total_number_words ,number_per_category, category_labels, filename ):
    print()
    print('Predicting labels')
    
    
    test_dataset['prediction'] = 'none'

    for index in test_dataset.index:
        conversation = word_tokenize( test_dataset['conversation'][index])
        len_conv = len (conversation)
        counter = FreqDist (conversation)
        arg_max = None
        prediction = None
        for category in category_labels:
            argument = 0

            number_words_per_feature = number_per_category[category]

            for word in conversation:

                argument += counter[word]/len_conv * log( (bag_of_words[category][word] + 1 )
                                                            /(number_words_per_feature + total_number_words) ) 

            argument += cat_prob [category]

            if arg_max == None:
                arg_max = argument
                prediction = category
            if arg_max < argument:
                arg_max = argument
                prediction = category

        test_dataset['prediction'][index] = prediction

    test_dataset.to_csv(filename , header=True, encoding='utf-8')

    return test_dataset


def results_val(test_dataset):
	from sklearn.metrics import confusion_matrix
    import itertools
    import matplotlib.pyplot as plt


    y_true = np.array (test_dataset['category'])
    y_pred = np.array(test_dataset['prediction'])

    classes = np.unique( test_dataset['category'] )

    confusion_matrix = confusion_matrix(y_true, y_pred)


    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    ##if normalize:
    ##    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    ##    print("Normalized confusion matrix")
    ##else:
    ##    print('Confusion matrix, without normalization')

 
    thresh = confusion_matrix.max() / 2.
    for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
        plt.text(j, i, confusion_matrix[i, j],
                 horizontalalignment="center",
                 color="white" if confusion_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.show()



##------------------------------------------------------------------
#                Main Function
##------------------------------------------------------------------

def main():
    #----Training DATASET
    #----PreProcess the training data
    train = pd.read_csv("trainining_dataset.csv", header=0)
    pre_processed_input = pre_process_words (train, 'Input_set_pre_processed.csv' )
    
    #----TEST DATASET
    #preprocessing the test set for making predictions
    test_set = pd.read_csv("test_input.csv", header=0)
    test_set = pre_process_words (test_set, 'Test_set_pre_processed.csv')


    #----TRAINED DATASET
    #train = pd.read_csv("Input_train.csv", header=0) 
    train_set, validation_set = train_test_split( train, test_size=0.2 ) 

    
    #----PARAMETERS OF THE TRAINED MODEL
    bag_words, total_number_words, number_per_feature = get_bag_words(train_set)
    category_labels = np.unique(input_dataset['category'])
    log_prob_cat = log_prob_category( train_set )

    #----PICKLING SOME FILES FOR RUNNING FASTER FURTHER FUNCTIONS
    ##pickle.dump (bag_words,open('bag_words_train.pickle','wb'))
    ##pickle.dump (total_number_words, open('total_n_words_train.pickle','wb'))
    ##pickle.dump (number_per_feature, open('number_words_p_feature_train.pickle','wb'))
    ##pickle.dump (log_prob_cat, open('log_prob_category_train.pickle','wb'))

    ##bag_words = pickle.load(open('bag_words_train.pickle','rb'))
    ##total_number_words = pickle.load(open('total_n_words_train.pickle','rb'))
    ##number_per_feature = pickle.load(open('number_words_p_feature_train.pickle','rb'))
    ##log_prob_cat = pickle.load(open('log_prob_category_train.pickle','rb'))


    ## -- PREDICTION OVER THE VALIDATION SET
    prediction_val = prediction(validation_set, bag_words, log_prob_cat ,total_number_words, number_per_feature,category_labels,'prediction_val_set.csv'  )
    results_val(prediction_val)
    print('prediction on the validation set - done')



    #----TRAINING IN THE WHOLE DATASET
    bag_words, total_number_words, number_per_feature = get_bag_words(input_dataset)
    
    category_labels = np.unique(input_dataset['category'])
    log_prob_cat = log_prob_category( input_dataset )

   
       #----PICKLING SOME FILES FOR RUNNING FASTER FURTHER FUNCTIONS
    pickle.dump (bag_words,open('bag_words_full.pickle','wb'))
    pickle.dump (total_number_words, open('total_n_words_full.pickle','wb'))
    pickle.dump (number_per_feature, open('number_words_p_feature_full.pickle','wb'))
    pickle.dump (log_prob_cat, open('log_prob_category_full.pickle','wb'))

    ##bag_words = pickle.load(open('bag_words_full.pickle','rb'))
    ##total_number_words = pickle.load(open('total_n_words_full.pickle','rb'))
    ##number_per_feature = pickle.load(open('number_words_p_feature_full.pickle','rb'))
    ##log_prob_cat = pickle.load(open('log_prob_category_full.pickle','rb'))

    ## -- PREDICTION OVER THE TEST SET
    prediction(test_set, bag_words, log_prob_cat ,total_number_words, number_per_feature, category_labels ,'prediction_test_set.csv' )
    

if __name__ == "__main__":
    main()
    print('done')