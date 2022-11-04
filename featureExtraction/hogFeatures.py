import cv2
import numpy as np

import dlib
from imutils import face_utils

import pandas as pd
import os
import math

from skimage import feature

from skimage.feature import hog


# Extract appearance features from images with HOG method
class HOGFeatures:

    
    df = pd.DataFrame(columns=[f'H{i}' for i in range(0, 7056)])
    df.insert(7056,'autism',1)
    df.insert(7057,'image_id',1)

    detector = dlib.get_frontal_face_detector()

    def calculateHOG(self, image):

        # - orientations: Number of bins in the histogram we want to create, the original research paper 
        #   used 9 bins so we will pass 9 as orientations
        # - pixels_per_cell: Determines the size of the cell. 
        #   The gradient matrices are divided into cells to form a block, we set to 8x8 cells.
        # - cells_per_block: Number of cells per block, will be 2x2 as mentioned previously. Ou seja, o "sliding" na imagem
        #   é feito passando um bloco 2x2 com 8x8 pixels, ou seja, 4 8x8 blocos no formato 2x2, é isso que é "sliding" pela imagem
        # - visualize: A boolean whether to return the image of the HOG, we set it to True so we can show the image.
        # - feature_vector: True or False. Return the data as a feature vector by calling .ravel() on the result just before returning.
        # - channel_axis: If None, the image is assumed to be a grayscale (single channel) image. 
        #   Otherwise, this parameter indicates which axis of the array corresponds to channels.
        # OUTPUTS:
        # - fd: HOG descriptor for the image. If feature_vector is True, a 1D (flattened) array is returned.
        # - hog_image: A visualisation of the HOG image. Only provided if visualize is True.
        fd = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False, 
                            feature_vector=True, channel_axis=None)

        return fd


    # category = ASD or TD 
    # autism = 1 or 0
    def calculateFeatures(self, category, autism):

        # Images to process
        directories = os.listdir(category)

        for direct in directories:

			# items inside directory
            items = os.listdir(category+"/"+direct+"/")
			# list ordering
            items = sorted(items, reverse=True)

            frame1 = None
            d = None

            for item in items:
                
                # current image path
                path = os.path.join(category+"/"+direct+"/"+item)
                
                frame = cv2.imread(path)
                
                # if neutral image, save as frame1
                if( item == "neutral.jpg" ):
                    frame1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                else:
                    frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    gray = frame2 - frame1
                    
                    fdHOG = self.calculateHOG(gray)
                    
                    fdHOG = fdHOG.tolist()

                    # join distances with autism and id_image columns
                    fdHOG.append(autism)
                    fdHOG.append(item)

                    # add row with result in dataframe
                    self.df.loc[len(self.df)] = fdHOG
                    

if __name__ == '__main__':
    
    hogFeatures = HOGFeatures()

    # calculate features in folder ASD
    hogFeatures.calculateFeatures("Dataset/ASD", 1)
    # calculate features in folder TD
    hogFeatures.calculateFeatures("Dataset/TD", 0)

    hogFeatures.df.to_csv("facialFeaturesHOG.csv")

