import sys
import numpy as np
import dlib
import cv2
import os.path
from featureValues import calcFeatureValues
from imutils import face_utils
import math
import shutil


# Find neutral and expressive images of each subject
class NeutralExpressive:
    
    predictor = None
    detector  = None

    def __init__(self):
        self.predictor = dlib.shape_predictor(os.path.join("preprocessing/shape_predictor_68_face_landmarks.dat"))
        self.detector  = dlib.get_frontal_face_detector()


    # Detect facial landmarks
    def detectLandmarks(self, img):
        
        # convert image to grayscale
        grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # face detect with a rectangle
        rects = self.detector(grayImage, 0)
        
        # initialize shape of current image
        shape = np.empty(68)
        
        # find the facial landmarks 
        for (i, face) in enumerate(rects):
            # make the prediction and transfom it to numpy array
            shape = self.predictor(grayImage, face)
            shape = face_utils.shape_to_np(shape)

        return shape


    # Calculate features for each person image 
    def getFeatures(self, folder_o, items):

        # list with the features of all images
        allFeatures, sumFeatures = [], [0]*14

        # for each item (image), start processing
        # calculate the feature values of each item of that person
        for item in items:
            
            # current image path
            path = os.path.join(folder_o+item)

            # detect facial landmarks
            img = cv2.imread(path)
            shape = self.detectLandmarks(img)

            try:
                # instance of feature values calculation
                FV = calcFeatureValues(shape)

                # list of feature values
                features = FV.getAllFeatureValues()
                featureValues = list(features.values())

                # adds the value of each feature to the total of that feature
                for index in range(len(featureValues)):
                    sumFeatures[index] += featureValues[index]

                # save feature values in a list
                allFeatures.append(featureValues)
                
            except:
                print("Unexpected error:", sys.exc_info()[0])

        return allFeatures, sumFeatures


    # Find expressive and neutral faces
    # Neutral Face: frame with the distance between the features closest to the average vector
    # Expressive Faces: the 10 frames with the biggest distance from the average vector
    def findNeutralExpressiveFaces(self, allFeatures, sumFeatures, items):
  
        # calculate the average of the features
        totalItems = len(allFeatures)
        averageFeatures = [x / totalItems for x in sumFeatures]

        # initialize variable with the shortest distance
        shortestDist, shortestDistItem = 100, ''

        # initialize the vector with the 10 largest distances
        greaterDist, greaterDistItems = [0]*10, ['']*10

        # iterate over the frames of the person
        for index in range(len(allFeatures)):

            # calculate the Euclidean distance of the average and current frame
            dist = math.dist(allFeatures[index], averageFeatures)

            # saves the smallest distance from the average (central frame)
            if( dist < shortestDist ):
                shortestDist = dist
                shortestDistItem = items[index]
            
            # find the index of the smallest value saved in the list
            minGreaterDist = min(greaterDist)
            indexMin = greaterDist.index(minGreaterDist)
            
            # check if the new distance is greater than the smallest distance that was saved in the list
            if(dist > minGreaterDist):
                # replace the item of that smallest distance in the list
                greaterDist[indexMin] = dist
                greaterDistItems[indexMin] = items[index]
        
        # return the shortest distance item and the greater distance items
        return shortestDistItem, greaterDistItems


    # Save neutral and expressive images in the destination folder
    def saveNeutralExpressiveImages(self, folder_o, folder_d):

        # get and sort items inside directory
        items = os.listdir(folder_o)
        items = sorted(items)

        # get features of a specific person/directory
        allFeatures, sumFeatures = self.getFeatures(folder_o, items)

        # find neutral and expressive faces
        shortestDistItem, greaterDistItems = self.findNeutralExpressiveFaces(allFeatures, sumFeatures, items)
        
        # copy the central frame
        shutil.copyfile(folder_o+shortestDistItem, folder_d+"neutral.jpg")

        # copy the expressive frames
        for items in greaterDistItems:
            shutil.copyfile(folder_o+items, folder_d+items)



if __name__ == '__main__':

    folder_o = "Teste-D3/"
    folder_d = "Teste-D4/"

    nE = NeutralExpressive()
    nE.saveNeutralExpressiveImages(folder_o, folder_d)
    