import numpy as np
import dlib
import cv2
import os.path
import pandas as pd
import sys
sys.path.append('preprocessing')
from featureValues import calcFeatureValues
from imutils import face_utils

# Extract features from images based on distance points
class GeometricFeatures:
	
	# create pandas dataframe to save data
	df = pd.DataFrame(columns=("ieb_height", "oeb_height", "eb_frowned", "eb_slanting", 
                               "eb_distance", "eeb_distance", "e_openness", "e_slanting", "m_openness", 
                               "m_mos", "m_width", "mul_height", "mll_height", "lc_height", "autism", "image_id"))

	# relative path
	my_path = ""

	# predictor and detector
	predictor, detector = None, None

	def __init__(self):
		self.my_path = os.path.abspath(os.path.dirname(__file__))
		self.predictor = dlib.shape_predictor(os.path.join("preprocessing/shape_predictor_68_face_landmarks.dat"))
		self.detector  = dlib.get_frontal_face_detector()


	# Detect facial landmarks
	def detectLandmarks(self, path):

		# read image
		image = cv2.imread(path)
        
		# Convert image to grayscale
		grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		# face detect with a rectangle
		rects = self.detector(grayImage, 0)

		# initialize shape of current image
		shape = np.empty(68)
		
		# For each detected face, find the landmark
		for (i, face) in enumerate(rects):

			# Make the prediction and transfom it to numpy array
			shape = self.predictor(grayImage, face)
			shape = face_utils.shape_to_np(shape)

		return shape


	# Calculate geometric features 
    # folder_o = source folder
    # autism = 1 (ASD) or 0 (TD)
	def calculateFeatures(self, folder_o, autism):

        # images to process
		directories = os.listdir(folder_o)

		# iterate over the directories
		for direct in directories:
            
			# list and ordering items inside directory
			items = os.listdir(folder_o+"/"+direct+"/")
			items = sorted(items, reverse=True)
			
			neutral_features = {}

			# for each item (image), start processing
			for item in items:
				
				# initialize dataframe with values = 0 
				row_dict = {
                    "ieb_height":0.0,  "oeb_height":0.0,   "eb_frowned":0.0, "eb_slanting":0.0, 
                    "eb_distance":0.0, "eeb_distance":0.0, "e_openness":0.0, "e_slanting":0.0, 
                    "m_openness":0.0,  "m_mos":0.0,        "m_width":0.0, 	  "mul_height":0.0, 
                    "mll_height":0.0,  "lc_height":0.0,    "autism": autism, "image_id":item
                }

				# current image path
				path = os.path.join(folder_o+"/"+direct+"/"+item)
				
				# detect facial landmarks
				shape = self.detectLandmarks(path)
			
				# instance of feature values calculation
				fV = calcFeatureValues(shape)

				# list of feature values
				features = fV.getAllFeatureValues()

                # if neutral face image, save shape and features
				if( item == "neutral.jpg" ):
					neutral_features = features
					
				else:
                
					for x in features:
                        # calculate difference between neutral and expressive features
						row_dict[x] = abs((features[x] - neutral_features[x]))

					self.df = self.df.append(row_dict, ignore_index=True)
			


if __name__ == '__main__':

	geoFeatures = GeometricFeatures()

	# calculate features in folder ASD
	geoFeatures.calculateFeatures("Dataset/ASD", 1)
	# calculate features in folder TD
	geoFeatures.calculateFeatures("Dataset/TD", 0)

	geoFeatures.df.to_csv("facialFeaturesExpressive.csv")
