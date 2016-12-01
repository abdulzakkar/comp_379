import os
from collections import Counter

def main():
    #Read in all the data
    os.chdir("C:/Users/Abdul/Documents/comp_379/cnn-text-classification-tf-master/cnn-text-classification-tf-master/data/rt-polaritydata")
    goodFile = open("rt-polarity.pos","r")
    goodReviews = goodFile.read()
    goodReviews = goodReviews.split("\n")
    
    badFile = open("rt-polarity.neg","r")
    badReviews = badFile.read()
    badReviews = badReviews.split("\n")
    
    os.chdir("C:/Users/Abdul/Documents/comp_379")
    removeFile = open("remove.txt","r")
    remove = removeFile.read()
    remove = remove.split("\n")
    
    ####################################################################################################
    #Getting all the words that appeat more than twice in the GOOD REVIEWS
    #Creating the training set
    ####################################################################################################
    
    temp = []
    allWords = []
    for i in range(0, int(len(goodReviews) * 0.7)):
        temp = goodReviews[i].split(" ");
        #split the review by the spaces
        temp = filter(lambda word: word not in remove, temp)
        #filter out unwanted words
        temp = temp[0:len(temp) - 1]
        for j in range(0, len(temp)):
            #iterate through all the words in the review
            allWords.append(temp[j])
            #add word to all words
    cGood = Counter(allWords)
    #get a count of all the words
    numGoodWords = len(allWords)  
    #Get the total words to later calculate probability of a word
    
    goodWords = []
    #the final set of words contains words that have appeared more than once in the training data.
    
    for key, value in cGood.iteritems():
        if value > 1:
            goodWords.append(str(key))
       
    ####################################################################################################
    #Getting all the words that appeat more than twice in the BAD REVIEWS
    #Creating the training set
    ####################################################################################################
       
    temp = []
    allWords = []
    for i in range(0, int(len(badReviews) * 0.7)):
        temp = badReviews[i].split(" ");
        temp = filter(lambda word: word not in remove, temp)
        temp = temp[0:len(temp) - 1]
        for j in range(0, len(temp)):
            allWords.append(temp[j])
    cBad = Counter(allWords)
    numBadWords = len(allWords)  

    badWords = []
    
    for key, value in cBad.iteritems():
        if value > 1:
            badWords.append(str(key))
    
    print("DEVELOPMENT SET")    
    
    ####################################################################################################
    #Creating and testing accuracy on the development set for GOOD REVIEWS
    ####################################################################################################
    
    developmentGoodReviews = goodReviews[int(len(goodReviews) * 0.7):int(len(goodReviews) * 0.85)]
   
    temp = []
    allWords = []
    countGoodCorrect = 0
    for i in range(0, len(developmentGoodReviews)):
        probGoodC = 1
        probBadC = 1
        temp = developmentGoodReviews[i].split(" ");
        temp = filter(lambda word: word not in remove, temp)
        temp = temp[0:len(temp) - 1]
        for j in range(0, len(temp)):
            if cGood.has_key(temp[j]) and cBad.has_key(temp[j]):
                probGoodC *= (float(cGood.get(temp[j])) / float(numGoodWords))
                probBadC *= (float(cBad.get(temp[j])) / float(numBadWords))
        if probGoodC > 0 and probBadC > 0 and probGoodC > probBadC:
            countGoodCorrect += 1

    print("\n\nAccuracy on GOOD REVIEWS: " + str((float(countGoodCorrect) / float(len(developmentGoodReviews))) * 100.) + "%")
    
    ####################################################################################################
    #Creating and testing accuracy on the development set for BAD REVIEWS
    ####################################################################################################
    
    developmentBadReviews = badReviews[int(len(badReviews) * 0.7):int(len(badReviews) * 0.85)]
   
    temp = []
    allWords = []
    countBadCorrect = 0
    for i in range(0, len(developmentBadReviews)):
        #iterate through all the reviews in the development section 
        probGoodC = 1
        probBadC = 1
        temp = developmentBadReviews[i].split(" ");
        temp = filter(lambda word: word not in remove, temp)
        #Split and filter like before
        temp = temp[0:len(temp) - 1]
        for j in range(0, len(temp)):
            #For each word in the review
            if cGood.has_key(temp[j]) and cBad.has_key(temp[j]):
                #if the word is in both the training sets (good AND bad reviews)
                probGoodC *= (float(cGood.get(temp[j])) / float(numGoodWords))
                #multiply the probability of this word with the probability of previous words.
                probBadC *= (float(cBad.get(temp[j])) / float(numBadWords))
        if probGoodC > 0 and probBadC > 0 and probGoodC < probBadC:
            #If the probability of the review being in the good reviews is LESS THAN
            #the probability of the review being in the bad reviews,
            countBadCorrect += 1
            #then we just predicted correctly, increase correctness count
                
            #                                                                       #
            #this calculation occurs for the other development set and the test sets#
            #                                                                       #
            
    print("Accuracy on BAD REVIEWS: " + str((float(countBadCorrect) / float(len(developmentBadReviews))) * 100.) + "%")
    
    ####################################################################################################
    #OVERALL DEVELOPMENT SET ACCURACY
    ####################################################################################################    
    
    print("Accuracy on ALL REVIEWS: " + str((float(countGoodCorrect + countBadCorrect) / float(len(developmentGoodReviews) + len(developmentBadReviews))) * 100.) + "%")
    
    print("\n\n------------------------------\n\nTEST SET")    
    
    ####################################################################################################
    #Creating and testing accuracy on the test set for GOOD REVIEWS
    ####################################################################################################
    
    testGoodReviews = goodReviews[int(len(goodReviews) * 0.85):int(len(goodReviews) * 1.)]
   
    temp = []
    allWords = []
    countGoodCorrect = 0
    for i in range(0, len(testGoodReviews)):
        probGoodC = 1
        probBadC = 1
        temp = testGoodReviews[i].split(" ");
        temp = filter(lambda word: word not in remove, temp)
        temp = temp[0:len(temp) - 1]
        for j in range(0, len(temp)):
            if cGood.has_key(temp[j]) and cBad.has_key(temp[j]):
                probGoodC *= (float(cGood.get(temp[j])) / float(numGoodWords))
                probBadC *= (float(cBad.get(temp[j])) / float(numBadWords))
        if probGoodC > 0 and probBadC > 0 and probGoodC > probBadC:
            countGoodCorrect += 1

    print("\n\nAccuracy on GOOD REVIEWS: " + str((float(countGoodCorrect) / float(len(testGoodReviews))) * 100.) + "%")
    
    ####################################################################################################
    #Creating and testing accuracy on the test set for BAD REVIEWS
    ####################################################################################################
    
    testBadReviews = badReviews[int(len(badReviews) * 0.85):int(len(badReviews) * 1.)]
   
    temp = []
    allWords = []
    countBadCorrect = 0
    for i in range(0, len(testBadReviews)):
        probGoodC = 1
        probBadC = 1
        temp = testBadReviews[i].split(" ");
        temp = filter(lambda word: word not in remove, temp)
        temp = temp[0:len(temp) - 1]
        for j in range(0, len(temp)):
            if cGood.has_key(temp[j]) and cBad.has_key(temp[j]):
                probGoodC *= (float(cGood.get(temp[j])) / float(numGoodWords))
                probBadC *= (float(cBad.get(temp[j])) / float(numBadWords))
        if probGoodC > 0 and probBadC > 0 and probGoodC < probBadC:
            countBadCorrect += 1
                
    print("Accuracy on BAD REVIEWS: " + str((float(countBadCorrect) / float(len(testBadReviews))) * 100.) + "%")
    
    ####################################################################################################
    #OVERALL TEST SET ACCURACY
    ####################################################################################################    
    
    print("Accuracy on ALL REVIEWS: " + str((float(countGoodCorrect + countBadCorrect) / float(len(testGoodReviews) + len(testBadReviews))) * 100.) + "%")
   
main()