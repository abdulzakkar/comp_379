import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model.perceptron import Perceptron


import time
import random
import sys
import os
import math
from graphics import *

#LOOK INTO KERNELIZED DISTANCE!!!!!!!!!!!!!




def printBoard(grid):
    for i in range(0,4):
        sys.stdout.write("|-------|-------|-------|-------|\n")
        for j in range(0,4):
            if grid[i][j] == 0:
                sys.stdout.write("|\t")
            else:
                sys.stdout.write("|" + str(grid[i][j]) + "\t")
        sys.stdout.write("|\n")
    sys.stdout.write("|-------|-------|-------|-------|\n")
    
def drawBoard(grid, win, tiles, numbers):   
    for i in range(0,4):
        for j in range(0,4):
            if (grid[j][i] != 0):
                hue = math.log(grid[j][i], 2) * 20
                x = round((1 - abs((float(hue) / float(60)) % float(2) - 1)) * 255)
                if hue >= 0 and hue < 60:
                    color = color_rgb(255, x, 0)
                elif hue >= 60 and hue < 120:
                    color = color_rgb(x, 255, 0)
                elif hue >= 120 and hue < 180:
                    color = color_rgb(0, 255, x)
                elif hue >= 180 and hue < 240:
                    color = color_rgb(0, x, 255)
                elif hue >= 240 and hue < 300:
                    color = color_rgb(x, 0, 255)
                elif hue >= 300 and hue < 360:
                    color = color_rgb(255, 0, x)
                tiles[i * 4 + j].setFill(color)
                numbers[i * 4 + j].setText(str(grid[j][i]))
            else:
                color = color_rgb(0,0,0)
                tiles[i * 4 + j].setFill(color)
                numbers[i * 4 + j].setText("")

def lose(grid):
    output = True
    for i in range(0,4):
        for j in range(0,4):
            if i < 3:
                if grid[i][j] == grid[i + 1][j]:
                    output = False
                    break
            if j < 3:
                if grid[i][j] == grid[i][j + 1]:
                    output = False
                    break
            if grid[i][j] == 0:
                output = False
                break
    return output

def shift(grid, move):
    rowcol = []
    if move % 2 == 0:
        #UP
        if int(move / 2) == 0:
            for i in range(0,4):
                multNext = True
                rowcol = []
                for j in range(0,4):
                    if grid[j][i] != 0:
                        if len(rowcol) == 0:
                            rowcol.append(grid[j][i])
                        else:
                            if rowcol[len(rowcol) - 1] != grid[j][i] or not multNext:
                                rowcol.append(grid[j][i])
                                multNext = True
                            else:
                                rowcol[len(rowcol) - 1] *= 2
                                multNext = False
                for j in range (0, 4 - len(rowcol)):
                    rowcol.append(0)
                for j in range(0,4):
                    grid[j][i] = rowcol[j]
        #DOWN
        else:
            for i in range(0,4):
                multNext = True
                rowcol = []
                for j in range(3,-1,-1):
                    if grid[j][i] != 0:
                        if len(rowcol) == 0:
                            rowcol.append(grid[j][i])
                        else:
                            if rowcol[len(rowcol) - 1] != grid[j][i] or not multNext:
                                rowcol.append(grid[j][i])
                                multNext = True
                            else:
                                rowcol[len(rowcol) - 1] *= 2
                                multNext = False
                for j in range (0, 4 - len(rowcol)):
                    rowcol.append(0)
                for j in range(0,4):
                    grid[3-j][i] = rowcol[j]
    else:
        #RIGHT
        if int(move / 2) == 0:
            for i in range(0,4):
                multNext = True
                rowcol = []
                for j in range(3,-1,-1):
                    if grid[i][j] != 0:
                        if len(rowcol) == 0:
                            rowcol.append(grid[i][j])
                        else:
                            if rowcol[len(rowcol) - 1] != grid[i][j] or not multNext:
                                rowcol.append(grid[i][j])
                                multNext = True
                            else:
                                rowcol[len(rowcol) - 1] *= 2
                                multNext = False
                for j in range (0, 4 - len(rowcol)):
                    rowcol.append(0)
                for j in range(0,4):
                    grid[i][3-j] = rowcol[j]
        #LEFT
        else:
            for i in range(0,4):
                multNext = True
                rowcol = []
                for j in range(0,4):
                    if grid[i][j] != 0:
                        if len(rowcol) == 0:
                            rowcol.append(grid[i][j])
                        else:
                            if rowcol[len(rowcol) - 1] != grid[i][j] or not multNext:
                                rowcol.append(grid[i][j])
                                multNext = True
                            else:
                                rowcol[len(rowcol) - 1] *= 2
                                multNext = False
                for j in range (0, 4 - len(rowcol)):
                    rowcol.append(0)
                for j in range(0,4):
                    grid[i][j] = rowcol[j]
    return grid

def spawn(grid):
    row = -1
    col = -1
    while row == -1 or col == -1 or grid[row][col] != 0:
        row = random.randint(0,3)
        col = random.randint(0,3)

    isTwo = random.randint(0,9)
    if isTwo == 0:
        grid[row][col] = 4
    else:
        grid[row][col] = 2
    return grid
    
def generateRandomGrid(grid):
    for i in range(0,4):
        for j in range(0,4):
            isTile = random.randint(0,1)
            if isTile == 0:
                grid[i][j] = 2**(random.randint(1,11))
            else:
                grid[i][j] = 0
    return grid

def tileToScore(n):
    tracker = 2
    count = 0
    while tracker != n:
        tracker *= 2
        count += 1
    return n * count

def calcScore(grid):
    score = 0
    for i in range(0,4):
        for j in range(0,4):
            if grid[i][j] != 0:
                score += tileToScore(grid[i][j])
    return score

def gridToData(grid):
    output = np.empty(0)
    maxTile = 0
    for i in range(0,4):
        for j in range(0,4):
            if grid[i][j] != 0:
                output = np.append(output, math.log(grid[i][j], 2))
            else:
                output = np.append(output, 0)
            if grid[i][j] > maxTile:
                maxTile = grid[i][j]
    return (output / math.log(maxTile, 2))
    
def gridToData2(grid):
    output = np.empty(0)
    maxTile = 0
    for i in range(0,4):
        for j in range(0,4):
            if grid[i][j] != 0:
                output = np.append(output, grid[i][j])
            else:
                output = np.append(output, 0)
    return output

def generateRandomWeights(bestWeights, radius):
    output = np.empty(0)
    for i in range(0,len(bestWeights)):
        output = np.append(output, random.randint(bestWeights[i]-radius,bestWeights[i]+radius))
    return output
    
def calculateDirection(weightList,grid):
    total = 0
    for i in range(0,4):
        for j in range(0,4):
            if (not grid[i][j] == 0):
                total += math.log(grid[i][j],2) * weightList[i * 4 + j]
    if weightList[16] > total:
        return 0
    else:
        return 1     

def main():
    while True:
        intro = Text(Point(250, 300), "2048 TRAINER\n\nTrain model...R\nFull game train...G\n\n>>>Delays between moves<<<\nTest KNN model...E\nTest Perceptron model...P\nRandom model...N\n\n>>>No delays<<<\nTest KNN model...F\nTest Perceptron model...S\nRandom model...M\n\nSimple learning model...L\n\nPRESS Q TO QUIT")
        intro.setSize(20)
        intro.setTextColor(color_rgb(255,255,255))
        
        
        if os.path.isfile("2048_train.csv"):
            data = pd.read_csv("2048_train.csv", header = None, usecols = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
            direction = pd.read_csv("2048_train.csv", header = None, usecols = [0])
            direction = np.transpose(direction)
            isPrevData = True
        else:
            isPrevData = False
            
        if os.path.isfile("best_weights.txt"):
            fileContent = open("best_weights.txt", 'r')
            weightString = fileContent.readlines()
            bestWeightsX = np.array(weightString[0:17])
            bestWeightsX = bestWeightsX.astype(np.float)
            bestWeightsY = np.array(weightString[17:34])
            bestWeightsY = bestWeightsY.astype(np.float)
            randomRadius = float(weightString[34])
            bestLearnScore = float(weightString[35])
            fileContent.close()
        else:
            bestWeightsX = np.empty(0)
            bestWeightsY = np.empty(0)
            for i in range(0,17):
                bestWeightsX = np.append(bestWeightsX,0)
                bestWeightsY = np.append(bestWeightsY,0)
                randomRadius = 100
                bestLearnScore = 0
            
        win = GraphWin("2048", 500, 600)  
        win.setBackground(color_rgb(0,103,105)) 
        intro.draw(win)       
            
        if isPrevData:
            options = ['r', 'g', 'e', 'n', 'f', 'm', 'l','q','s','p']
        else:
            options = ['r', 'g', 'n', 'm', 'l','q']
        mode = '-'
        while mode not in options:
            mode = win.getKey()
            
        if mode == 'q':
            win.close()
            return 0
            
        if mode == 'e' or mode == 'f':
            knn = KNeighborsClassifier(n_neighbors = 10)
            knn.fit(data, np.ravel(np.transpose(direction)))

        if mode == 'p' or mode == 's':
            ppn = Perceptron(eta0=0.01, n_iter=10000)
            ppn.fit(data, np.ravel(np.transpose(direction)))
            
        intro.undraw()
        win.setBackground(color_rgb(100,100,100))   
        
        score = 0
        scoreText = Text(Point(250, 550), str(score))
        scoreText.setTextColor(color_rgb(255,255,255))
        scoreText.setSize(30)
        
        board = np.array([[0,0,0,0],
                          [0,0,0,0],
                          [0,0,0,0],
                          [0,0,0,0]])
    
        isChangeBoard = np.array([[0,0,0,0],
                                  [0,0,0,0],
                                  [0,0,0,0],
                                  [0,0,0,0]])
          
        #DO THE SAME THING FOR THE TILE NUMBERS                        
        tileList = []
        numberList = []
    
        board = spawn(board)
        board = spawn(board)
        
        for i in range(0,4):
            for j in range(0,4): 
                tileList.append(Rectangle(Point(i * 125 + 5, j * 125 + 5), Point(i * 125 + 120, j * 125 + 120)))
                numberList.append(Text(Point(i * 125 + 60, j * 125 + 60), str(board[j][i])))
                numberList[i * 4 + j].setSize(20)
                tileList[i * 4 + j].setWidth(4)
                tileList[i * 4 + j].setOutline(color_rgb(255,255,255))
                numberList[i * 4 + j].setTextColor(color_rgb(0,0,0))
                tileList[i * 4 + j].draw(win)
                numberList[i * 4 + j].draw(win)
        
        #THE TRAINING DATA
        if not isPrevData:
            data = np.empty((0,16), float)
            direction = np.empty(0)

        drawBoard(board, win, tileList, numberList) 
    
        move = '-'
        classes = ['w','s','d','a']    
        moveIter = 0   
        nMoves = 0
        
        totalIterations = 1
        iteration = 1
        
        currentWeightsX = generateRandomWeights(bestWeightsX, math.floor(randomRadius))
        currentWeightsY = generateRandomWeights(bestWeightsY, math.floor(randomRadius))
        currentBestWeightsX = currentWeightsX
        currentBestWeightsY = currentWeightsY
        
        if mode == 'l':
            maxIterations = int(input("Enter number of generations to simulate:"))
        
        while True:
    
            score = calcScore(board)
    
            if mode != 'l':
                drawBoard(board, win, tileList, numberList)
            
            scoreText.undraw()
            if mode != 'l':
                scoreText.setSize(30)
                scoreText.setText(str(score))
                scoreText.draw(win)
    
            for i in range(0,4):
                for j in range(0,4):
                    isChangeBoard[i][j] = board[i][j]
    
            if mode == 'r' or mode == 'g':
                move = win.getKey()
                if nMoves % 30 == 0 and mode == 'r':
                    board = generateRandomGrid(board)
                nMoves += 1
            else:
                if mode == 'e' or mode == 'n' or mode == 'p':
                    time.sleep(0.5)
                if mode == 'e' or mode == 'f':
                    probs = np.ravel(knn.predict_proba(gridToData(board).reshape(1,-1)))
                    ranks = [0] * len(probs)
                    for i, x in enumerate(sorted(range(len(probs)), key=lambda y: probs[y])):
                        ranks[x] = i
                    move = classes[ranks[moveIter]]
                    moveIter += 1
                elif mode == 'm':
                    move = classes[random.randint(0,3)]
                elif mode == 'l':
                    move = classes[calculateDirection(currentWeightsX,board) * 2 + 
                        calculateDirection(currentWeightsY,board)]
                    if moveIter > 0:
                        move = classes[random.randint(0,3)]
                    moveIter += 1
                elif mode == 'p' or mode == 's':
                    move = ppn.predict(gridToData(board).reshape(1,-1))
                    if moveIter > 0:
                        move = classes[random.randint(0,3)]
                    moveIter += 1

            if move == 'w':
                board = shift(board, 0)
            elif move == 'd':
                board = shift(board, 1)
            elif move == 's':
                board = shift(board, 2)
            elif move == 'a':
                board = shift(board, 3)
            elif move == 'q':
                break
                
            if lose(board):
                print score
                if mode == 'l':
                    board = np.array([[0,0,0,0],
                                      [0,0,0,0],
                                      [0,0,0,0],
                                      [0,0,0,0]])
    
                    isChangeBoard = np.array([[0,0,0,0],
                                              [0,0,0,0],
                                              [0,0,0,0],
                                              [0,0,0,0]])
    
                    board = spawn(board)
                    board = spawn(board)
                    
                    scoreText.setSize(16)
                    scoreText.setText("Score: " + str(score) + " -- iteration: " + str(iteration) + " -- r: " + str(randomRadius))                    
                    
                    if score > bestLearnScore:
                        bestLearnScore = score
                        currentBestWeightsX = currentWeightsX
                        currentBestWeightsY = currentWeightsY
                        print "current best" + str(currentBestWeightsX) + str(currentBestWeightsY)
                      
                    if iteration == 1000:
                        drawBoard(board, win, tileList, numberList)
                        print "best score: " + str(bestLearnScore)
                        iteration = 1
                        randomRadius *= 0.9
                        bestWeightsX = currentBestWeightsX
                        bestWeightsY = currentBestWeightsY
                        print "best weights:" + str(bestWeightsX) + str(bestWeightsY)
                        if totalIterations / 1000 >= maxIterations:
                            toWrite = np.empty(0)
                            toWrite = np.append(toWrite,bestWeightsX)
                            toWrite = np.append(toWrite,bestWeightsY)
                            toWrite = np.append(toWrite,randomRadius)
                            toWrite = np.append(toWrite,bestLearnScore)
                            np.savetxt("best_weights.txt",toWrite)
                            #WRITE WEIGHTS N STUFF
                            break
                        
                    totalIterations += 1
                    iteration += 1
                    
                    currentWeightsX = generateRandomWeights(bestWeightsX, math.floor(randomRadius))
                    currentWeightsY = generateRandomWeights(bestWeightsY, math.floor(randomRadius))
                else:
                    break
            
            isChanged = False
            for i in range(0,4):
                for j in range(0,4):
                    if isChangeBoard[i][j] != board[i][j]:
                        isChanged = True        
            
            if isChanged:
                moveIter = 0
                if mode == 'r' or mode == 'g':
                    direction = np.append(direction, move)
                    data = np.vstack([data, gridToData2(board)])
                board = spawn(board)
                
        score = calcScore(board)
    
        scoreText.setSize(16)
        scoreText.setText(str(score) + " -- YOU LOSE (Press Q)")
        while move != 'q':
            move = win.getKey()
        win.close()
        
        if mode == 'r' or mode == 'g':
            pd.DataFrame.to_csv(pd.DataFrame(np.hstack([np.reshape(direction, [np.shape(direction)[0], 1]), data])), "2048_train.csv", index = False, header = False)

main()