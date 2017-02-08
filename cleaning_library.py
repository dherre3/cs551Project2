"""
Created on Sat Feb  4 13:31:39 2017

@author: davidherrera
"""
import numpy as np
import re

def cleanWordsLessThanThree(string):
    arrayOfWords = string.split( )
    arrayOfWords = np.array([word for word in arrayOfWords if len(word)>3])
    string = ' '.join(arrayOfWords)
    return string

pattern = '<[^>]*>'
prog = re.compile(pattern)
def cleanAngleBrackets(x):
    x = re.sub(prog,'',x)
    return x

def cleanData(string):

    string = cleanAngleBrackets(string)
    string = cleanWordsLessThanThree(string)
    return string