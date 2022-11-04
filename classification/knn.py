import pandas as pd
import sys
import numpy as np

sys.path.append('Utils')

from distances import Distances as Dist

#######################################################
#                                                     #
#   k-Nearest Neighborhood Algorithm                  #
#                                                     #
#   Receives an integer value as k and a string value #
#   as the name of distance.                          #
#                                                     #
#######################################################
class Knn:

    k = 1
    distance = "euclidean"
    X_train = None
    y_train = None

    # k: distance list size 
    # distance: distance used
    def __init__(self, k, distance):
        self.k = k
        self.distance = distance


    # X_train: features matrix 
    # y_train: label vector
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train 


    # predict new data
    def predict(self, data):

        # generate list of k data with shortest distances 
        distanceList = self.generateDistanceList(data)

        # labels counters
        countClass0 = 0
        countClass1 = 0

		# loop on shortest distances list 
        for elem in distanceList:
            # get label of data
            label = self.y_train.iloc[elem[0]]
            # check and increment label counter
            if( label == 1.0 ):
                countClass1 += 1
            elif( label == 0.0 ):
                countClass0 += 1

		# return the class with more data in shortest distance list
        if( countClass1 > countClass0 ):
            return 1.0
        else:
            return 0.0

    # predict dataset
    def predictAll(self, X_test):

        result = []
        # loop on test data
        for (i,rowTest) in X_test.iterrows():
            result.append( self.predict(rowTest) )

        return result

    # method to find the highest value on the list 
    # receives a list with the distances in the format: [data index, distance] 
    # return the tuple of the list with the highest distance value 
    def greaterDistance(self, distList):
        maior = distList[0][1]
        indexMaior = 0

        for i in range(len( distList ) ):
            if( distList[i][1] > maior ):
                maior = distList[i][1]
                indexMaior = i

        return (indexMaior, maior)

    # method to generate the list of distances 
    # newData: list with feature values
    def generateDistanceList(self, newData):

        distanceList = []

        # Measure the distance of the new data with train data 
		# loop on train dataset
        count = 0
        for (i,rowTrain) in self.X_train.iterrows():

            # convert Series to List
            rowTrain = rowTrain.tolist() 

            # compute the manhattan distance between the analyzed row and the new data 
            if( self.distance == "manhattan" ):
                dist = 1 - Dist.manhattanDistance(newData, rowTrain)
            elif( self.distance == "cosine" ):  
                dist = 1 - Dist.cosineSimilarity(newData, rowTrain)
            else:
                dist = 1 - Dist.euclideanDistance(newData, rowTrain)
            
            # check if the list is already at maximum size, if not, just add new data to the list
            if(len(distanceList) < self.k ):
                distanceList.append([count, dist]) 
                
            # otherwise, add only if the distance is smaller
            else:
                # check which is the biggest distance value 
                (posMaior, maiorDist) = self.greaterDistance(distanceList)

                # check if dist is less than the largest value in the list
                if( dist < maiorDist ): 
                    # replace the value on list
                    distanceList[posMaior] = [count, dist]	

            count+=1
        return distanceList