# Autism Detection

This repository contains the codes of a project developed during a master's research.

Autism Spectrum Disorder (ASD) is a neuro-developmental disability marked by deficits in communicating and interacting with others. The standard protocol for diagnosis is based on fulfillment of descriptive criteria, which does not establish precise measures.

This repository contains the algorithms implemented to collect and filter a new facial features dataset (https://data.mendeley.com/datasets/j47m3nw4mc) of ASD and Typical Development (TD) subjects. In addition, it contains the classification algorithms used to investigate the patterns and differences between the groups (ASD and TD).

The project methodology consists of the following: 

1. Video Selection
2. Data Preprocessing
3. Feature Extraction
4. Classification Methods
5. Performance Analysis

## Video Selection

The videos were manually selected from YouTube. 
The frames were extracted (fps=1) using FFmPEG software. 
The dataset collected cannot yet be made publicly available.

## Data Preprocessing

Since the videos were from YouTube and in a wild context, it was necessary to filter and remove some frames. 
The preprocessing applied to the frames extracted consists filter the images based on the following:

- Face detection
- Pose variation: Pose Estimation; Face Alignment
- Selection of neutral and expressive frames

The folder *preprocessing* contains the algorithms implemented for this purpose. 
Face Detection and Face Alignment steps are made in the *detectAlignFace.py* algorithm.
Pose Variation step is made in the *poseEstimation.py* algorithm.
Selection of neutral and expressive frames is made in the *neutralExpressiveFace.py* algorithm.

Note: The experiments were also made analyzing just the eye region. In this case, the algorithm *CropEye.py* iterates over the neutral and expressive images and crops the eye region.


## Feature Extraction

After the previous step, we have 1 neutral frame and 10 expressive frames of each person.
To extract the features from the images was used algorithms available in the sklearn library.


## Classification Methods

To classify the data, it was implemented classical Artificial Intelligence (AI) classification methods from scratch. 
They were implemented from scratch to guarantee better experiment control and personal knowledge. 
