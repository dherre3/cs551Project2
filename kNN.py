#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 18:01:41 2017

@author: davidherrera
"""

import numpy as np
import heapq
import collections

def getTopNNeighbours(n, distanceVector,x_output):
    heap = []
    final = []
    outputCategories = []
    #build initial heap  
    for distance in range(n):
        heapq.heappush(heap, (-distanceVector[distance],distance))
    for distance in range(n,len(distanceVector)):
        maxVal = heapq.heappop(heap)
        if maxVal[0] < -distanceVector[distance]:
            heapq.heappush(heap,maxVal)
            heapq.heapreplace(heap,(-distanceVector[distance],distance))
        else:
            heapq.heappush(heap,maxVal)
    for neighbor in range(n):
        final.append((heap[neighbor][1],-heap[neighbor][0],x_output[heap[neighbor][1]]))
        outputCategories.append(x_output[heap[neighbor][1]])
    return (final,outputCategories)
 
def getMajorityClass(topNneighbors):
    frequency_count = collections.Counter(topNneighbors)
    return frequency_count.most_common(1)[0][0]
    
def getEucledianDistance(x_train, x_predict_sample):
    distance = np.zeros(x_train.shape[0])
    for i in range(x_train.shape[0]):
        distance[i] = np.sqrt(np.sum(np.square(x_train[i] - x_predict_sample)))
    return distance
def kNN(x_train, y_train,x_predict,number_neighbors=3):
    y_predict = []
    distance = np.zeros(x_predict.shape[0])
    for sample in range(x_predict.shape[0]):
        print("\nSample " +str(sample+1) +" OF "+ str(x_predict.shape[0]))
        #distance = np.sqrt(np.sum(np.square(x_train - x_predict[sample]),axis=1))
#        print(x_train[0].shape,x_predict[sample].shape)
#        break;
        distance = getEucledianDistance(x_train,x_predict[sample])
        #print("======Obtaining top 10 neighbors")
        (distanceTuples, topCategories) = getTopNNeighbours(number_neighbors,distance,y_train)
        #print("Getting Majority Class")
        #print(distanceTuples)
        y_predict.append(getMajorityClass(topCategories))
        print("Done Getting Majority class: "+y_predict[sample])
    return y_predict
             

            
        
        
    
    