# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# This python script uses dlib facial landmark predictor and a custom trained gaze direction classifier     #
# in order to work. I created my own dataset to train the gaze direction classifier with eye images that are#
# 72x10 resolution.                                                                                         #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import cv2
import numpy as np
import dlib
import time
import pandas as pd
from math import hypot
import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
from pynput.keyboard import Key, Controller

# The network used to train the classifier
# Network structer should be changed for different models
class Net(Module):   
    def __init__(self):
        super(Net, self).__init__()

        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
            Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            
            
            Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
        )
        #self.dropout = Dropout(0.4) 
        self.linear_layers = Sequential(
            Linear(36, 4)
        )

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x
        
def midpoint(p1,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

#calculate blingking ratio

def isBlink(p1,p2,p3,p4):
    hor_line_length = hypot((p1[0]-p2[0]),(p1[1]-p2[1]))
    ver_line_length = hypot((p3[0]-p4[0]),(p3[1]-p4[1]))
    ratio = hor_line_length/ver_line_length
    # ratio maybe different for every person
    if ratio<0.16:
        return True
    else:
        return False

# Function for calculating the most frequent direction in the buffer
# According to the direction a keyboard inputs are commented out

def countRegion(buffer):
    print(buffer)
    arr = np.zeros(5,dtype=int)
    for i in range(5):
        if buffer[i] == b'a':
            arr[0] = arr[0] + 1
        elif buffer[i] == b'd':
            arr[1] = arr[1] + 1
        elif buffer[i] == b'w':
            arr[2] = arr[2] + 1
        elif buffer[i] == b's':
            arr[3] = arr[3] + 1
        elif buffer[i] == b'n':
            arr[4] = arr[4] + 1    
    indexMax = np.argmax(arr)
    if indexMax == 4:
        #keyboard.press(Key.space)
        #keyboard.release(Key.space)
        print("MID")
    elif indexMax == 0:
        #keyboard.press('a')
        #keyboard.release('a') 
        print("LEFT")
    elif indexMax == 1:
        #keyboard.press('d')
        #keyboard.release('d')
        print("RIGHT")
    elif indexMax == 2:
        #keyboard.press('w')
        #keyboard.release('w')
        print("TOP")
    elif indexMax == 3:
        #keyboard.press('s')
        #keyboard.release('s')
        print("BOT")
    
# Function for inputing each captured image into the CNN
# According to the confidence level, corresponding buffer index is incremented

def predictRegion(eyeArrL,eyeArrR,buffer,index):
    eyeArrL = cv2.resize(eyeArrL,(36,10))
    eyeArrR = cv2.resize(eyeArrR,(36,10))
    eye_concat = np.concatenate((eyeArrL,eyeArrR),axis=1)
    eye_concat = eye_concat.astype(np.float32)
    eye_concat /= 255.0
    eye_concat  = torch.from_numpy(eye_concat)
    eye_concat = eye_concat.reshape(1, 1, 10, 72)
    with torch.no_grad():
        output = model(eye_concat)
        softmax = torch.exp(output).cpu()
        prob = list(softmax.numpy())
        predictions = np.argmax(prob, axis=1)
        maxval = np.amax(prob)
    #print(maxval)
    if(maxval<14.00 and maxval>0.05):
        buffer[index] = 'n'
    elif(predictions == 0):
        buffer[index] = 'a'
    elif(predictions == 1):
        buffer[index] = 'd'
    elif(predictions == 2):
        buffer[index] = 'w'
    elif(predictions == 3):
        buffer[index] = 's'
    #time.sleep(1)
    return buffer


cap = cv2.VideoCapture(0)
# dlib facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
count = 0
columns = ["left","right"]
df = pd.DataFrame(columns=columns)
keyboard = Controller()
model = Net()
# Pre trained pytorch model
model.load_state_dict(torch.load("gaze_direction_predictor4.pt"))
model.eval()
eye_cropR = np.zeros((36,10),dtype=float)
eye_cropL = np.zeros((36,10),dtype=float)
buffer = np.chararray(5)
while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    faces = detector(gray)
    if count==4:
        countRegion(buffer)
        count=0
        buffer = np.chararray(5)
        buffer[:] = ''
    for face in faces:
        count+=1
        landmarks = predictor(gray, face)
        left_p1 = (landmarks.part(36).x, landmarks.part(36).y)
        right_p1 = (landmarks.part(39).x, landmarks.part(39).y)
        eyetop1 =  midpoint(landmarks.part(37),landmarks.part(38))
        eyebot1 = midpoint(landmarks.part(41),landmarks.part(40))
        left_p2 = (landmarks.part(42).x, landmarks.part(42).y)
        right_p2 = (landmarks.part(45).x, landmarks.part(45).y)
        eyetop2 = midpoint(landmarks.part(43),landmarks.part(44))
        eyebot2 = midpoint(landmarks.part(47),landmarks.part(46))
        right_eye_region = np.array([(landmarks.part(36).x, landmarks.part(36).y),
                                    (landmarks.part(37).x, landmarks.part(37).y),
                                    (landmarks.part(38).x, landmarks.part(38).y),
                                    (landmarks.part(39).x, landmarks.part(39).y),
                                    (landmarks.part(40).x, landmarks.part(40).y),
                                    (landmarks.part(41).x, landmarks.part(41).y)], np.int32)
        cv2.polylines(frame, [right_eye_region],True,(0,0,255),2)
       
        left_eye_region = np.array([(landmarks.part(42).x, landmarks.part(42).y),
                                    (landmarks.part(43).x, landmarks.part(43).y),
                                    (landmarks.part(44).x, landmarks.part(44).y),
                                    (landmarks.part(45).x, landmarks.part(45).y),
                                   (landmarks.part(46).x, landmarks.part(46).y),
                                   (landmarks.part(47).x, landmarks.part(47).y)], np.int32)
        
        cv2.polylines(frame, [left_eye_region],True,(0,0,255),2)     
        
        height, width, _ = frame.shape
        mask = np.zeros((height, width), np.uint8)
        mask1 = np.zeros((height, width), np.uint8)
        cv2.polylines(mask, [right_eye_region], True, 255, 2)
        cv2.fillPoly(mask, [right_eye_region], 255)
        
        right_eye = cv2.bitwise_and(gray, gray, mask=mask)
        eye_cropR = right_eye[eyetop1[1]:eyebot1[1],left_p1[0]:right_p1[0]]
        
        dims = eye_cropR.shape
        
        cv2.polylines(mask1, [left_eye_region], True, 255, 2)
        cv2.fillPoly(mask1, [left_eye_region], 255)
        left_eye = cv2.bitwise_and(gray, gray, mask=mask1)
        eye_cropL = left_eye[eyetop2[1]:eyebot2[1],left_p2[0]:right_p2[0]]
        dimsL = eye_cropL.shape
        
        for j in range(dims[0]):
            for i in range(dims[1]):
                if eye_cropR[j][i] == 0:
                    eye_cropR[j][i] = 255
          
        for j in range(dimsL[0]):
            for i in range(dimsL[1]):
                if eye_cropL[j][i] == 0:
                    eye_cropL[j][i] = 255    
         #historgram equalization
        eye_cropR = cv2.equalizeHist(eye_cropR)
        eye_cropL = cv2.equalizeHist(eye_cropL)
        #pd.DataFrame([eye_cropR,eye_cropL]).to_csv("test.csv")
        #Işık durumuna göre cv2.threshold değeri değiştirilmeli
        _ ,thresh_right = cv2.threshold(eye_cropR,75,255,cv2.THRESH_BINARY)
        _ ,thresh_left = cv2.threshold(eye_cropL,75,255,cv2.THRESH_BINARY)
        

        #thres_right = cv2.dilate(thresh_right,(3,3),iterations = 1) zz
        thresh_right = cv2.morphologyEx(thresh_right, cv2.MORPH_CLOSE, (3,3))
        thresh_left = cv2.morphologyEx(thresh_left, cv2.MORPH_CLOSE, (3,3))
        

        if isBlink(eyetop2,eyebot2,left_p2,right_p2):    #LEFT EYE
            #keyboard.press('z')
            #keyboard.release('z')
            #time.sleep(1)
            print("BLINKING")
            continue
        if isBlink(eyetop1,eyebot1,left_p1,right_p1):   #RIGHT EYE
            #keyboard.press('z')
            #keyboard.release('z')
            #time.sleep(1)
            print("BLINKING")
            continue        
        buffer = predictRegion(eye_cropL,eye_cropR,buffer,count)
        #temp = getRegion(thresh_right,thresh_left)
        #buffer[count] = temp
        thresh_right = cv2.resize(thresh_right,None,fx=10,fy=10)
        thresh_left = cv2.resize(thresh_left,None,fx=10,fy=10)
    
    tmpR = cv2.resize(eye_cropR,None,fx=5,fy=5)
    tmpL = cv2.resize(eye_cropL,None,fx=5,fy=5)
    cv2.imshow("Right_eye",tmpR)
    cv2.imshow("Left_eye",tmpL)
    cv2.imshow("boundary",frame) 
    key = cv2.waitKey(1)
    # STOP KEY ESC
    if key == 27:
        break    
    
cap.release()
cv2.destroyAllWindows()