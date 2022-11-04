import cv2
import numpy as np
import dlib
from imutils import face_utils
import pandas as pd
import os
import math
from skimage import feature


# Extract appearance features from images with LBP method
class lbp:

    df = pd.DataFrame(columns=(
        "histBin1","histBin2","histBin3","histBin4","histBin5","histBin6",
        "histBin7","histBin8","histBin9","histBin10","histBin11","histBin12",
        "histBin13","histBin14","histBin15","histBin16","histBin17","histBin18",
        "histBin19","histBin20","histBin21","histBin22","histBin23","histBin24",
        "histBin25","histBin26",
        "autism", "image_id"
     ))

    detector = dlib.get_frontal_face_detector()


    def calculateLBP(self, grayImage):

        # Capture details at such a small scale is the biggest drawback to the original LBP algorithm 
        # To handle variable neighborhood sizes, an extension to the original LBP implementation was proposed by Ojala et al. 
        # Two parameters were introduced:
        # - The number of points p in a circularly symmetric neighborhood to consider 
        #   (thus removing relying on a square neighborhood).
        # - The radius of the circle r, which allows us to account for different scales.
        radius = 3
        numPoints = 8 * radius

        # When surrounding pixels are all black or all white, then that image region is flat. 
        # Groups of continuous black or white pixels are considered 'uniform' patterns that can be 
        # interpreted as corners or edges.
        # - method: uniform indicates that we are computing the rotation and grayscale invariant form of LBPs
        # - numPoints: Number of circularly symmetric neighbour set points (quantization of the angular space).
        # - radius: Radius of circle (spatial resolution of the operator).
        # - output: lbp image
        lbp = feature.local_binary_pattern(grayImage, numPoints, radius, method="uniform")

        return lbp


    def calculateHistogram(self, lbp, numPoints):

        # compute the lbp histogram 
        # this functon counts the number of times each of the LBP prototypes appears
        # the range and bins are relative to possible values and to the uniform method
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, numPoints + 3), range=(0, numPoints + 2))

        # normalize the histogram
        hist = hist.astype("float")
        eps=1e-7
        hist /= (hist.sum() + eps)

        return hist




    # category: ASD or TD 
    # autism: 1 or 0
    def calculateFeatures(self, category, autism):

        # images to process
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

                    lbp = self.calculateLBP(gray)
                    hist = self.calculateHistogram(lbp, 24)
                    
                    hist = hist.tolist()

                    # join distances with autism and id_image columns
                    hist.append(autism)
                    hist.append(item)

                    # add row with result in dataframe
                    self.df.loc[len(self.df)] = hist
            

if __name__ == '__main__':

    lbpFeatures = lbp()
		
    # calculate features in folder ASD
    lbpFeatures.calculateFeatures("Dataset/ASD", 1)
    # calculate features in folder TD
    lbpFeatures.calculateFeatures("Dataset/TD", 0)

    lbpFeatures.df.to_csv("facialFeaturesLBP.csv")

