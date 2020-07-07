# gazetrack-classifier

This is a Python script that classifies 5 different gaze directions(left, right, middle, top, bottom) and blinking. The input image is taken from a camera connected to the PC and each image frame is
fed into 2 different machine learning models. First model being dlib facial landmark predictor(included) and second the being a custom CNN model that I created and trained with a
dataset consisting only of my face(not a good idea). Will add the data generation and CNN training scripts in the future.

REQUIREMENTS:

- cv2
- numpy
- pytorch
- dlib
- pandas
