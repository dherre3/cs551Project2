'''
Author: David Herrera
Date: 02/02/2017
'''

import re
pattern = '<[^>]*>'
prog = re.compile(pattern)

def cleanAngleBrackets(x):
    x = re.sub(prog,'',x)
    return x