#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 22:28:11 2020

@author: daniela
"""
import os
import numpy as np
from cv2 import cv2
import matplotlib.pyplot as plt

#%%
def resizer(img, height):
    
    ratio = img.shape[1]/img.shape[0]
    width = round(ratio*height)
    return cv2.resize(img, (width, height))

#%%
N = 8

inputPath = os.path.abspath('../dataset/video/vid'+str(N)+'.mov')
outputPath = os.path.abspath('output/vidSet'+str(N))
height = 500

#%%
cap = cv2.VideoCapture(inputPath)

w, h = cap.get(3), cap.get(4)
numFrame = (int(cap.get(7)))
variance = np.zeros(numFrame)
for n in range(numFrame):
    ret,frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    variance[n] = cv2.Laplacian(frame, cv2.CV_64F).var()

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
#%%
minDist = 5
window = 15
modulo = numFrame % window
K = int((numFrame-modulo)/window)
frameList = np.zeros(K+1)
#plt.plot(variance)
#plt.show()

for i in range(K+1):
    
    if i == K:
        #print(str(K*window+1),str(K*window+modulo))
        currFrame = variance[K*window+1:K*window+modulo-1]
    else:
        #print(str(i*window+1),str((i+1)*window))
        currFrame = variance[i*window+1:(i+1)*window]
    
    frameList[i] = np.where(variance==np.max(currFrame))[0]
    
    if i > 0 and i != K:
        
        while frameList[i]-frameList[i-1] <= minDist:
            
            currFrame = np.sort(currFrame)
            currFrame = currFrame[::-1]
            currFrame = currFrame[1:];
            frameList[i] = np.where(variance==currFrame[0])[0]
            
#%%
if os.path.exists(outputPath):
    fileList = [f for f in os.listdir(outputPath)]
    for f in fileList:
        os.remove(os.path.join(outputPath,f))
else: os.mkdir(outputPath)

cap = cv2.VideoCapture(inputPath)

for i in frameList:
    cap.set(1, int(i))
    ret, frame = cap.read()
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    frameName = str('frame_'+ str(int(i))+ '.png')
    cv2.imwrite(os.path.join(outputPath, frameName), frame)
    
cap.release()

