#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import glob
import numpy as np
from cv2 import cv2
import copy
import sys


def resizer(img, height, toResize):
    '''ratio = width/height'''
    if toResize == False:
        return img
    ratio = img.shape[1]/img.shape[0]
    width = round(ratio*height)
    return cv2.resize(img, (width, height))

def computeHomography(im1, im2):
    ''' '''
    sift = cv2.SIFT_create()
    # find the keypoints with SIFT
    kp1, des1 = sift.detectAndCompute(im1, None)
    kp2, des2 = sift.detectAndCompute(im2, None)

    des1 = np.float32(des1)
    des2 = np.float32(des2)
    # create flann basedMatcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Match descriptors.
    matches = flann.knnMatch(des1,des2,k=2)

    #good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)

    return H

def makeOrder(images,centered):
    
    numIm = len(images)
    imOrder = list(range(numIm))
    if centered:
        centIm = int(np.floor(numIm/2))
        imOrder1 = list(range(centIm,-1,-1))
        imOrder2 = list(range(centIm+1,numIm))
        imOrder = imOrder1 + imOrder2
        print(numIm,imOrder1, imOrder2)
    return imOrder

def makeOrderHomograpy(imOrder, images):
    hMat = []
    hMat.append(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
    for i in range(0,len(imOrder)-1):
        H = computeHomography(images[imOrder[i+1]], images[imOrder[i]])
        hMat.append(H)

    for i in range(1,len(hMat)):
        hMat[i] = hMat[i].dot(hMat[i-1])

    return hMat

def outputLimits(hMat, imShape1, imShape2):

    h1, w1 = imShape1
    h2, w2 = imShape2
    pts1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    pts2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    pts2_ = cv2.perspectiveTransform(pts2, hMat)
    pts = np.concatenate((pts1, pts2_), axis=0)
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)

    return xmin, xmax, ymin, ymax

def sizeCheck(xmin, ymin, xmax, ymax):
    ''' '''
    thresh = 2500
    if (xmin > thresh) or (ymin > thresh) or (xmax > thresh) or (ymax > thresh):
        print('image too large')
        print('xmin:' + str(xmin) + ' | ' + 'ymin:' + str(ymin) + ' | ' +
              'xmax:' + str(xmax) + ' | ' + 'ymax:' + str(ymax))
        sys.exit()
    else:
        return

def warp2Images(dst, src, H,t):
    
    [xmin, xmax, ymin, ymax] = outputLimits(H, dst.shape[:2], src.shape[:2])
    sizeCheck(xmin, ymin, xmax, ymax)
 
    Ht = np.array([[1, 0, t[1]], [0, 1, t[0]], [0, 0, 1]])
    src_warped = cv2.warpPerspective(src, Ht.dot(H), (dst.shape[1],dst.shape[0]),
                                     cv2.BORDER_TRANSPARENT)
    
    src_warped[src_warped==0] = dst[src_warped==0]
    return src_warped

def createPano(hMat, imOrder,imgRGB):
    
    limits = np.zeros((len(images),4))
    for i in range(0, len(images)-1):
        limits[i] = outputLimits(hMat[i+1], imgRGB[imOrder[i]].shape[:2],
                                 imgRGB[imOrder[i+1]].shape[:2])

    xmin = round(np.min(limits[:,0]))
    xmax = round(np.max(limits[:,1]))
    ymin = round(np.min(limits[:,2]))
    ymax = round(np.max(limits[:,3]))

    pano = imgRGB[imOrder[0]]
    pad_widths = [-ymin, max(ymax, pano.shape[0]) - pano.shape[0],
                  -xmin, max(xmax, pano.shape[1]) - pano.shape[1]]
    pano_pad = cv2.copyMakeBorder(pano, *pad_widths, cv2.BORDER_CONSTANT)

    t = [-ymin,-xmin]
    for i in range(1,len(hMat)):
        pano_pad = warp2Images(pano_pad,imgRGB[imOrder[i]], hMat[i],t)
        cv2.imshow(' ', pano_pad)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return pano_pad
#%%
## TODO: Poznamka: tu si zavadis takovy hlavni config, odkud se co bude brat a
# parametry, co budes pouzivat, ale pod timto configem mas hned funkci
# (ted jednu, pozdeji vice), hned pak musis scrollovat daleko od configu,
# abys mohla napriklad kontrolovat prubeh skriptu - doporucil bych mit:
    # vsechny potrebne importy
    # vsechny vyuzivane funkce
    # config pro zadavani menicich se parametru
    # samotny prubeh (behem ktereho se volaji ty funkce)

N = 1
testSet = ['set0', 'set1', 'set2', 'set3','set4','set5','set6', 'set7', 'set8']
#inputPath = os.path.abspath('data/' + testSet[N] + '/*.png')
#inputPath = os.path.abspath('../dataset/foto/persp/'+testSet[N]+'/*.jpeg')
inputPath = os.path.abspath('../dataset/old/Memorial_Hall/*.jpg')
outputPath = os.path.abspath('output/test' +testSet[N]+'/*.JPG')

centered = True
resize = True
HEIGHT = 500
#%%
imList = sorted(glob.glob(inputPath))

# load images
images = []
imgRGB = []
for i in imList:
    i = cv2.imread(i)
    i = resizer(i, HEIGHT, resize)
    imgRGB.append(i)
    images.append(cv2.cvtColor(i, cv2.COLOR_BGR2GRAY))


imOrder = makeOrder(images,centered)
hMat = makeOrderHomograpy(imOrder, images)
cv2.imwrite('pano.jpg', createPano(hMat, imOrder, imgRGB))


