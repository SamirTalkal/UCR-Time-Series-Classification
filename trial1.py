# -*- coding: utf-8 -*-
"""
Created on Wed May 15 01:35:31 2019

@author: Samir Talkal
"""

import numpy as np
import pandas as pd


full_train_data = pd.read_csv('CBF_TRAIN.csv', header = None)
full_test_data = pd.read_csv('CBF_TEST.csv', header = None)

train_dataset = full_train_data.iloc[:,range(1,full_train_data.shape[1])] #traindatawithoutlabel
label_train = full_train_data.iloc[:,0] #trainlabel

test_dataset = full_test_data.iloc[:,range(1,full_test_data.shape[1])] #testdatawithoutlabel
label_test = full_test_data.iloc[:,0] #testlabel

train_dataset = train_dataset.values
label_train = label_train.values

test_dataset=test_dataset.values
label_test=label_test.values

#Classification_algorithm

count=0
def class_algo(traindata,trainlabel,classobj):
    best_case = float("inf")
    predicted_class = 0
    global count
    count+=1
    for j in range(1,len(trainlabel)):
        compare_the_obj = traindata[j,:]
        dist = np.sqrt(np.sum((np.array(compare_the_obj)-np.array(classobj))**2)) #Euclidean distance formula
        if dist < best_case:
            predicted_class = trainlabel[j]
        best_case = dist
    return predicted_class 


correct = 0

for k in range(1,len(label_test)):
    classification_obj_1 = test_dataset[k,:]
    actual_class = label_test[k]
    predicted_class = class_algo(train_dataset,label_train,classification_obj_1)
    if predicted_class == actual_class:
        correct =correct+1
print("Correct Predicted class: {0} , accuracy:{1} ".format(correct,correct/count))
