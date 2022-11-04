import cv2
import numpy as np

import dlib
from imutils import face_utils

import pandas as pd
import os
import math


# Extract motion features from images with Optical Flow method
class sparseOpticalFlow:

    df = pd.DataFrame(columns=(
        "leftEyePts43","leftEyePts44","leftEyePts45","leftEyePts46","leftEyePts47","leftEyePts48",
        "rightEyePts37","rightEyePts38","rightEyePts39","rightEyePts40","rightEyePts41","rightEyePts42",
        "nosePts28","nosePts29","nosePts30","nosePts31","nosePts32","nosePts33","nosePts34","nosePts35","nosePts36",
        "mouthPts49","mouthPts50","mouthPts51","mouthPts52","mouthPts53","mouthPts54","mouthPts55","mouthPts56",
        "mouthPts57","mouthPts58","mouthPts59","mouthPts60","mouthPts61","mouthPts62","mouthPts63","mouthPts64",
        "mouthPts65","mouthPts66","mouthPts67","mouthPts68",
        "leftEyebrowPts18","leftEyebrowPts19","leftEyebrowPts20","leftEyebrowPts21","leftEyebrowPts22",
        "rightEyebrowPts23","rightEyebrowPts24","rightEyebrowPts25","rightEyebrowPts26","rightEyebrowPts27",

        "autism", "image_id"
     ))
    
    # relative path
    my_path = ""

	# predictor and detector
    predictor, detector = None, None

    def __init__(self):
        self.my_path = os.path.abspath(os.path.dirname(__file__))
        self.predictor = dlib.shape_predictor(os.path.join("preprocessing/shape_predictor_68_face_landmarks.dat"))
        self.detector  = dlib.get_frontal_face_detector()


    # calculate the distance between the points
    def distanceFlow(self, p0, p1):

        distances = []

        for i in range(len(p0[0])):
            x1 = np.int32(p0[0][i][0])
            y1 = np.int32(p0[0][i][1])

            x2 = np.int32(p1[0][i][0])
            y2 = np.int32(p1[0][i][1])
            
            # euclidean distance
            result = ((x1 - x2)**2) + ((y1 - y2)**2)

            distances.append(math.sqrt(result))
        
        return distances
        

    # Detect facial landmarks
    # select some points and return them
    def facialLandmarks(self, grayImage):

        # face detect with a rectangle
        rects = self.detector(grayImage, 0)

        result = []
        # initialize shape of current image
        shape = np.empty(68)
        
        # For each detected face, find the landmark.
        for (i, face) in enumerate(rects):

            # Make the prediction and transfom it to numpy array
            shape = self.predictor(grayImage, face)
            shape = face_utils.shape_to_np(shape)
            shape = np.float32(shape)


        leftEyePts = shape[42:48]
        rightEyePts = shape[36:42]
        nosePts = shape[27:36]
        mouthPts = shape[48:68]
        leftEyebrowPts = shape[17:22]
        rightEyebrowPts = shape[22:27]
        
        shapeSelected = np.concatenate([leftEyePts, rightEyePts, nosePts, mouthPts, leftEyebrowPts, rightEyebrowPts])
        
        result.append(shapeSelected)
        arr = np.array(result)
        
        return arr



    # category = ASD or TD 
    # autism = 1 or 0
    def calculateFeatures(self, category, autism, lk_params):

        # Images to process
        directories = os.listdir(category)

        for direct in directories:
            
			# items inside directory
            items = os.listdir(category+"/"+direct+"/")
			# list ordering
            items = sorted(items, reverse=True)

            p0 = None
            frame1 = None

            count = 0      

            for item in items:
                
                # current image path
                path = os.path.join(category+"/"+direct+"/"+item)
                
                frame = cv2.imread(path)
                
                # if neutral image, calculate facial landmarks
                if( item == "neutral.jpg" ):
                    frame1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    p0 = self.facialLandmarks(frame1)
                    count += 1
                else:
                    frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    # calculate optical flow
                    p1, st, err = cv2.calcOpticalFlowPyrLK(frame1, frame2, p0, None, **lk_params)
                    
                    # calculate the distance between the points
                    distance = self.distanceFlow(p0, p1)
                    
                    # join distances with autism and id_image columns
                    distance.append(autism)
                    distance.append(item)

                    # add row with result in dataframe
                    self.df.loc[len(self.df)] = distance
            

if __name__ == '__main__':

    opticalFlow = sparseOpticalFlow()

    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15, 15),
                maxLevel = 2,
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # calculate features in folder ASD
    opticalFlow.calculateFeatures("Dataset/ASD", 1, lk_params)
    # calculate features in folder TD
    opticalFlow.calculateFeatures("Dataset/TD", 0, lk_params)

    opticalFlow.df.to_csv("facialFeaturesOpticalFlow.csv")
