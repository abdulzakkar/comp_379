# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 22:53:38 2016

@author: Abdul Zakkar
"""

from sklearn import linear_model
import pandas as pd
import numpy as np
       
def logisticRegressionTest(filePath, isRefined):
    handWrittenData = pd.read_csv(filePath, header = None)
    #Read in data:
    #THE DATA SET I USED:
    #Data contains x and y evenly spaced coordinates for the natural handwriting of numbers.
    #16 columns: x1 y1 x2 y2 x3 y3 etc., then the last column is the number written.    
    
    classes = handWrittenData.iloc[0:,16].values 
    #The numbers written are saved in this variable.
    
    classesTrain = handWrittenData.iloc[0:int(0.7 * len(classes)),16].values
    #The numbers written that will be used for TRAINING are in this variable.  
    
    dataTrain = handWrittenData.iloc[0:int(0.7 * len(classes)),0:16].values
    #The coordinates that will be used for TRAINING are in this variable.     
    
    classesTest = handWrittenData.iloc[int(0.7 * len(classes)):,16].values
    #The numbers written that will be used for TESTING are in this variable.     
    
    dataTest = handWrittenData.iloc[int(0.7 * len(classes)):,0:16].values
    ##The coordinates that will be used for TESTING are in this variable.      
    if isRefined:
        logistic = linear_model.LogisticRegression(C = 100.0, tol = 0.00001, solver = "newton-cg").fit(dataTrain, classesTrain)
        print("\n\n>>>Logistic Regression Score: Parameters Refined")
    else:
        logistic = linear_model.LogisticRegression().fit(dataTrain, classesTrain)
        print("\n\n>>>Logistic Regression Score: Parameters Default")
    #Use the logistic regression model.
    
    print(logistic.score(dataTest, classesTest))
    #Score the model.
    
    print("\nA few tests:")
    #Ensure the model is working properly.
    
    print("\nData associated with the handwritten number " + str(classesTest[0]) + " predicted as...")
    print(logistic.predict(dataTest[0,0:16].reshape(1, -1)))
   
    print("\nData associated with the handwritten number " + str(classesTest[1]) +  " predicted as...")
    print(logistic.predict(dataTest[1,0:16].reshape(1, -1)))
    
    print("\nData associated with the handwritten number " + str(classesTest[2]) +  " predicted as...")
    print(logistic.predict(dataTest[2,0:16].reshape(1, -1)))

def myKNearestNeighbor(filePath, kVal):
    #Same data setup as Logistic Regression (I know it's inefficient)
    handWrittenData = pd.read_csv(filePath, header = None) 
    classes = handWrittenData.iloc[0:,16].values 
    classesTrain = handWrittenData.iloc[0:int(0.7 * len(classes)),16].values
    dataTrain = handWrittenData.iloc[0:int(0.7 * len(classes)),0:16].values
    classesTest = handWrittenData.iloc[int(0.7 * len(classes)):,16].values
    dataTest = handWrittenData.iloc[int(0.7 * len(classes)):,0:16].values
    
    nCorrect = 0
    #Keeps track of how many correct predictions occurred.
    for i in range(0, len(classesTest)):
        #iterate through every sample set in the test data...
        distances = np.zeros(len(classesTrain))
        #create an array of distances between this sample and all the training samples.
        for j in range(0, len(classesTrain)):
            #iterate through all of the training samples...
            total = 0
            #keeps track of distance calculation.
            for k in range(0, len(dataTest[0,])):
                #iterate through each data point in the sample to compare with the training sample.
                #Uses extended version of the pythagoras' theorem to calculate distances.
                total += (dataTest[i][k] - dataTrain[j][k])**2
            distances[j] = float(total)**(1/2.0)
        indexes = np.argpartition(distances, kVal)[:kVal]
        #gathers indexes of smallest distances.
        if np.bincount(classesTrain[indexes]).argmax() == classesTest[i]:
        #most common training sample class is found, does it match the test sample class?
            nCorrect += 1
        if i % 100 == 0:
            print(str(i) + " of " + str(len(classesTest)) + " complete.")
            #Progress display (because it's REALLY slow)
    print("K Nearest Neighbor accuracy @ k = " + str(kVal) + " is " + str((float(nCorrect)/float(len(classesTest))) * 100) + "%.")
    #calculates accuracy.
            
       
def main():
    #PROBLEM 1
    logisticRegressionTest("C:/Users/Abdul/Documents/comp_379/handwritten_digits_data.csv", False)
    
    #PROBLEM 2
    logisticRegressionTest("C:/Users/Abdul/Documents/comp_379/handwritten_digits_data.csv", True)
    
    #The second parameter:
        #True: use refined parameters
        #False: use default parameters
    
    #PROBLEM 3
    myKNearestNeighbor("C:/Users/Abdul/Documents/comp_379/handwritten_digits_data.csv", 5)
    myKNearestNeighbor("C:/Users/Abdul/Documents/comp_379/handwritten_digits_data.csv", 10)
    
main()

