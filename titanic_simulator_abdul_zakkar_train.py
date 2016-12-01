import csv
import os
from scipy import stats
import numpy

#Returns information on the regression line formed by numerical variables (e.g. fare)
def continuousVarLearn(dictionary):
    percentList = []
    valueList = []
    for key, value in sorted(dictionary.iteritems()):
        percentList.append(float(value.count('1')) / float(len(value))) 
        #each numerical value gets a percentage of survival. These points are plotted
        #to obtain a regression line.
        valueList.append(key)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(valueList, percentList)
    return abs(r_value), slope, intercept

#Returns information on likelihood of survival of each item in categorical variables (e.g. Sex)
def categoryVarLearn(dictionary):
    percentList = []
    categoryList = []
    for key, value in dictionary.iteritems():
        percentList.append(float(value.count('1')) / float(len(value)))
        #Survival percentages are found by compiling survivals of all memebers of a given item
        categoryList.append(key)
    stdCorrected = numpy.std(percentList) * 2
    return percentList, categoryList, stdCorrected

def main():
    #open training file
    os.chdir("C:\Users\Abdul\Documents\comp_379")
    inputFile = open("train.csv","rb")
    reader = csv.reader(inputFile)
    
    #numerical variables:
    survivalPerAge = {}
    survivalPerSibSp = {}
    survivalPerParCh = {}
    survivalPerFare = {}
    survivalPerClass = {}
    
    #categorical variables:
    survivalPerGender = {}
    survivalPerEmbark = {}
    
    #Go through all test data and put all information into the dictionaries above.
    #Dictionary format:     key: survivals
    #E.g.                   male: 10101010010100110110110101010
    
    inputRatioSurvived = 0
    inputRatioTotal = 0
    for row in reader:
        r = row
        for i in range(2,len(r)):
            if i == 2 and r[i] != "Pclass" and r[i] != "":
                if survivalPerClass.get(float(r[i])) == None:
                    survivalPerClass[float(r[i])] = r[1]
                else:
                    survivalPerClass[float(r[i])] += r[1]
            elif i == 4 and r[i] != "Sex" and r[i] != "":
                if survivalPerGender.get(r[i]) == None:
                    survivalPerGender[r[i]] = r[1]
                else:
                    survivalPerGender[r[i]] += r[1]
            elif i == 5 and r[i] != "Age" and r[i] != "":
                if survivalPerAge.get(float(r[i])) == None:
                    survivalPerAge[float(r[i])] = r[1]
                else:
                    survivalPerAge[float(r[i])] += r[1]
            elif i == 6 and r[i] != "SibSp" and r[i] != "":
                if survivalPerSibSp.get(float(r[i])) == None:
                    survivalPerSibSp[float(r[i])] = r[1]
                else:
                    survivalPerSibSp[float(r[i])] += r[1]
            elif i == 7 and r[i] != "Parch" and r[i] != "":
                if survivalPerParCh.get(float(r[i])) == None:
                    survivalPerParCh[float(r[i])] = r[1]
                else:
                    survivalPerParCh[float(r[i])] += r[1]
            elif i == 9 and r[i] != "Fare" and r[i] != "":
                if survivalPerFare.get(float(r[i])) == None:
                    survivalPerFare[float(r[i])] = r[1]
                else:
                    survivalPerFare[float(r[i])] += r[1]
            elif i == 11 and r[i] != "Embarked" and r[i] != "":
                if survivalPerEmbark.get(r[i]) == None:
                    survivalPerEmbark[r[i]] = r[1]
                else:
                    survivalPerEmbark[r[i]] += r[1]
        if r[1] == '1':
            inputRatioSurvived += 1
        inputRatioTotal += 1
    
    inputFile.close()
    inputRatio = float(inputRatioSurvived)/float(inputRatioTotal)
    
    #Each dictionary is used to create a probability model for each variable,
    #either using regression or simple probabilities in the case of categorical variables.           
    ageRValue, ageSlope, ageIntercept = continuousVarLearn(survivalPerAge)
    sibSpRValue, sibSpSlope, sibSpIntercept = continuousVarLearn(survivalPerSibSp)
    parChRValue, parChSlope, parChIntercept = continuousVarLearn(survivalPerParCh)
    fareRValue, fareSlope, fareIntercept = continuousVarLearn(survivalPerFare)
    classRValue, classSlope, classIntercept = continuousVarLearn(survivalPerClass)
    
    genderPercents, genderCategories, genderStd = categoryVarLearn(survivalPerGender)
    embarkPercents, embarkCategories, embarkStd = categoryVarLearn(survivalPerEmbark)
    
    #open testing file
    inputFile = open("train.csv","rb")
    reader = csv.reader(inputFile)
    
    #open output file
    outputFile  = open('train_complete.csv', "wb")
    writer = csv.writer(outputFile, delimiter=',')
    
    header = True   

    finalResultList = []
    passengerIDList = []
    
    for row in reader:
        r = row
        
        if not header:
            #weights are used to give certain variables more importance.
            #weights decided through r coefficient for numerical values,
            #or standard deviation for categorical values.
            sumOfWeights = 0
            
            if r[2] != "Pclass" and r[2] != "":
                sumOfWeights += classRValue
            if r[4] != "Sex" and r[4] != "":
                sumOfWeights += genderStd
            if r[5] != "Age" and r[5] != "":
                sumOfWeights += ageRValue
            if r[6] != "SibSp" and r[6] != "":
                sumOfWeights += sibSpRValue
            if r[7] != "Parch" and r[7] != "":
                sumOfWeights += parChRValue
            if r[9] != "Fare" and r[9] != "":
                sumOfWeights += fareRValue
            if r[11] != "Embarked" and r[11] != "":
                sumOfWeights += embarkStd
                
            finalResult = 0
            
            #The final result is calculated by taking a weighted average of all the 
            #variables' probability predictions.
            if r[2] != "Pclass" and r[2] != "":
                finalResult += (classRValue/sumOfWeights) * (classSlope * float(r[2]) + classIntercept)
            if r[4] != "Sex" and r[4] != "":
                finalResult += (genderStd/sumOfWeights) * (genderPercents[genderCategories.index(r[4])])
            if r[5] != "Age" and r[5] != "":
                finalResult += (ageRValue/sumOfWeights) * (ageSlope * float(r[5]) + ageIntercept)
            if r[6] != "SibSp" and r[6] != "":
                finalResult += (sibSpRValue/sumOfWeights) * (sibSpSlope * float(r[6]) + sibSpIntercept)
            if r[7] != "Parch" and r[7] != "":
                finalResult += (parChRValue/sumOfWeights) * (parChSlope * float(r[7]) + parChIntercept)
            if r[9] != "Fare" and r[9] != "":
                finalResult += (fareRValue/sumOfWeights) * (fareSlope * float(r[9]) + fareIntercept)
            if r[11] != "Embarked" and r[11] != "":
                finalResult += (embarkStd/sumOfWeights) * (embarkPercents[embarkCategories.index(r[11])]) 
            
            print "Result:", finalResult
            
            finalResultList.append(finalResult)
            passengerIDList.append(r[0])
        else:
            header = False
            
    cutOff = finalResultList[int(round(inputRatio * float(len(finalResultList))))]
    writer.writerow(["PassengerId","Survived"])
    for i in range(0,len(finalResultList)): 
        if finalResultList[i] > cutOff:
            writer.writerow([passengerIDList[i],'1'])
        else:
            writer.writerow([passengerIDList[i],'0'])
    
    inputFile.close()
    outputFile.close()
      
main()