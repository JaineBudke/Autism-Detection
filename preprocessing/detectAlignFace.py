import os
import sys
import dlib
import cv2
import numpy as np
from imutils import face_utils


# Check if there is a face in the image. If true, align the face. 
class DetectAlignFace:
    
    predictor = None
    detector  = None

    def __init__(self):
        self.predictor = dlib.shape_predictor(os.path.join("preprocessing/shape_predictor_68_face_landmarks.dat"))
        self.detector  = dlib.get_frontal_face_detector()


    # Align the face based on the center of the eyes
    # Code based on: https://www.pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/ 
    def alignFace(self, image, leftEyePts, rightEyePts):
        
        # compute the center of mass for each eye
        leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
        rightEyeCenter = rightEyePts.mean(axis=0).astype("int")

        # compute the angle between the eye centroids
        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180

        # compute the desired right eye x-coordinate based on the
        # desired x-coordinate of the left eye
        desiredLeftEye=(0.35, 0.35)
        desiredRightEyeX = 1.0 - desiredLeftEye[0]

        # determine the scale of the new resulting image by taking
        # the ratio of the distance between eyes in the current image
        # to the ratio of distance between eyes in the desired image
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desiredDist = (desiredRightEyeX - desiredLeftEye[0])

        desiredFaceHeight = 125
        desiredFaceWidth  = 125

        desiredDist *= desiredFaceWidth
        scale = desiredDist / dist

        # compute center (x, y)-coordinates (i.e., the median point)
        # between the two eyes in the input image
        eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
            (leftEyeCenter[1] + rightEyeCenter[1]) // 2)

        # convert to integer
        eyesCenter = ( int(eyesCenter[0]), int(eyesCenter[1]) ) 
        
        # grab the rotation matrix for rotating and scaling the face
        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)
        
        # update the translation component of the matrix
        tX = desiredFaceWidth * 0.5
        tY = desiredFaceHeight * desiredLeftEye[1]
        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])

        # apply the affine transformation
        (w, h) = (desiredFaceWidth, desiredFaceHeight)
        output = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)
        
        # return the aligned face
        return output
    

    # Save the aligned image in the destination folder 
    def saveAlignedFace(self, folder_o, folder_d, path):
        
        try:
            
            # read the image
            image = cv2.imread(folder_o+path)
            
            # crop the image
            grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            rects = self.detector(grayImage, 0)

            # for each detected face, find the landmark
            for (i, face) in enumerate(rects):
            
                # estimate facial landmark points and transfom it to numpy array
                shape = self.predictor(grayImage, face)
                shape = face_utils.shape_to_np(shape)

                leftEyePts = shape[42:48]
                rightEyePts = shape[36:42]

                output = self.alignFace(image, leftEyePts, rightEyePts)

                cv2.imwrite(folder_d+path, output)
                    
        except:
            print("Unexpected error:", sys.exc_info()[0])
        

if __name__ == '__main__':
    
    faceFilter = DetectAlignFace()

    # source folder
    folder_o = "Teste-O/"
    # destination folder
    folder_d = "Teste-D/"

    # get images to process
    images_path = os.listdir(folder_o)

    # for each image, extract and align the face, and save in the destination folder
    for img_path in images_path:
        faceFilter.saveAlignedFace(folder_o, folder_d, img_path)