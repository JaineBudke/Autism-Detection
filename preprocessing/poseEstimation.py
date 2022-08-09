import cv2
import numpy as np
import dlib
from imutils import face_utils
import os
import sys


# Estimate the position of the face in relation to the camera
class PoseEstimation:

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


    # Get some 2D and 3D points and camera coordinates for the algorithm 
    def getParametersValues(self, img):
        
        size = img.shape
        
        # detect the face
        shape = self.detectLandmarks(img)
        
        # 2D image points
        points_2D = np.array([
                                shape[30],     # Nose tip 
                                shape[8],      # Chin     
                                shape[36],     # Left eye left corner 
                                shape[45],     # Right eye right corner
                                shape[48],     # Left Mouth corner 
                                shape[54]      # Right mouth corner
                                ], dtype="double")

        # 3D model points
        points_3D = np.array([
                                (0.0, 0.0, 0.0),             # Nose tip
                                (0.0, -330.0, -65.0),        # Chin
                                (-225.0, 170.0, -135.0),     # Left eye left corner
                                (225.0, 170.0, -135.0),      # Right eye right corne
                                (-150.0, -150.0, -125.0),    # Left Mouth corner
                                (150.0, -150.0, -125.0)      # Right mouth corner
                                ])

        # Camera internals
        focal_length = size[1]
        center = (size[1]/2, size[0]/2)
        camera_matrix = np.array([[focal_length, 0, center[0]],
                                    [0, focal_length, center[1]],
                                    [0, 0, 1]], dtype = "double")

        
        return points_2D, points_3D, camera_matrix


    # Estimate the face position and save the images that pass through the filter
    def estimate(self, folder_o, folder_d, subThresh):
    
        try:
            
            # read Image
            img = cv2.imread(folder_o)
            
            # get algorithm parameters 
            points_2D, points_3D, camera_matrix = self.getParametersValues(img)
            dist_coeffs = np.zeros((4,1))
            
            # apply solvePnP to obtain rotation and translation vectors
            success, rotation_vector, translation_vector = cv2.solvePnP(points_3D, points_2D, camera_matrix, dist_coeffs, flags=0) 
            
            # project 3D point to 2D image using the vectors obtained 
            nose_end_point2D, jacobian = cv2.projectPoints(np.array([(0.0, 0.0, 500.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
            
            # define start and end points that represent the projection  
            point1 = ( int(points_2D[0][0]), int(points_2D[0][1]))
            point2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
            
            # define a vector with the two points by subtracting them
            subP = (abs(point1[0]-point2[0]), abs(point1[1]-point2[1]))
            
            # filter the images based on the threshold 
            if(subP[0] <= subThresh):
                # save the image if it passes through the filter
                cv2.imwrite(folder_d, img)
                
        except:
            print("Unexpected error:", sys.exc_info()[0])
                


if __name__ == '__main__':

    faceFilter = PoseEstimation()

    # source folder
    folder_o = "Teste-D/"
    # destination folder
    folder_d = "Teste-D2/"

    # get images to process
    images_path = os.listdir(folder_o)

    # for each image, estimate the face position, 
    # and save it in the destination folder if it is less than the threshold
    for img_path in images_path:
        
        subThresh = 30.0
        faceFilter.estimate(folder_o+img_path, folder_d+img_path, subThresh)
