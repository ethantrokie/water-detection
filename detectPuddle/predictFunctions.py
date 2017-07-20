import cv2
import numpy as np
import Imagetransformations
import helperFunc
import features
import random
import os
import containerFunctions as ct
from sklearn.externals import joblib
import math

def moduleC(vidpath,maskpath,outputFolder, numFrames, dFactor, densityMode,boxSize,NumbofFrameSearch,numbofSamples,patchSize,numFramesAvg):
    preprocess = ct.preprocessVideo(vidpath,numFrames,dFactor,densityMode)
    helperFunc.saveVid(preprocess, outputFolder + vidpath[-12:-4] + "residual_most_recent.avi")
    features, isWater = getFeatures(preprocess,maskpath,dFactor,boxSize,NumbofFrameSearch,numbofSamples,patchSize,numFramesAvg)
    return features, isWater
def getFeatures(preprocessedVid,maskpath,dscale,boxSize,TemporalLength,numbofSamples,patchSize,numFramesAvg):
    #sets up vars
    width = preprocessedVid.shape[1]
    height = preprocessedVid.shape[0]

    #water == True in mask
    mask = features.createMask(maskpath,dscale)
    mask = mask[:,:,0]

    #size of edge around the image-- will later crop this part out
    minrand = max(int(boxSize / 2 + 1), int(patchSize / 2))

    #obtain temporal feature and corp the image
    temporalFeat = features.fourierTransformFullImage(preprocessedVid,boxSize,mask)
    temporalFeat[0:minrand,:] = 0
    temporalFeat[:,0:minrand] = 0
    temporalFeat[height-minrand:height, :] = 0
    temporalFeat[:,width-minrand:width] = 0
    spaceFeat = features.SpatialFeaturesFullImage(preprocessedVid,patchSize,numFramesAvg)
    combinedFeature = np.concatenate((temporalFeat,spaceFeat),2)


    print("finished computing unified featureSpace")
    isWater = mask.astype(np.uint8)
    combinedFeature = combinedFeature.astype(np.float32)
    return combinedFeature, isWater



def testFullVid(vidpath,maskpath,outputFolder, numFrames, dFactor, densityMode,boxSize,numbofFrameSearch,numbofSamples,patchSize,numFramesAvg):
    feature, trueMask = moduleC(vidpath,maskpath,outputFolder,numFrames,dFactor,densityMode,boxSize,numbofFrameSearch,numbofSamples,patchSize,numFramesAvg)
    width = feature.shape[1]
    height = feature.shape[0]
    kernel = np.ones((5, 5), np.float32)
    minrand = max(int(boxSize / 2 + 1), int(patchSize / 2))
    # load the SVM model
    model = cv2.ml.SVM_load("just_ponds.xml")
    isWaterFound = np.zeros((height,width),  dtype = np.int)
    probabilityMask = np.zeros((height,width),  dtype = np.float64)
    placeHodler = 0
    for i in range(height):
        for j in range(width):
            h = feature[i,j,:]
            h = h.reshape(-1, h.shape[0])
            dist = model.predict(h,placeHodler,1)[1]
            prob = 1/(1+math.exp(-1*dist))
            probabilityMask[i, j] = 1 - prob
            if(prob < 0.5):
                isWaterFound[i,j] = True
            else:
                isWaterFound[i,j] = False
        print(i)

    isWaterFound = isWaterFound.astype(np.uint8)
    isWaterFound = isWaterFound[minrand:height-minrand, minrand:width-minrand]
    trueMask = trueMask[minrand:height-minrand,minrand:width-minrand]
    isWaterFound[isWaterFound == 1] = 255
    trueMask[trueMask == 1] = 255
    beforeReg = outputFolder + vidpath[-12:-4] + '_before_regularization' + '.png'
    cv2.imwrite(beforeReg, isWaterFound)
    for i in range(11):
        isWaterFound = regularizeFrame(isWaterFound,probabilityMask,.0001)
    #isWaterFound = cv2.morphologyEx(isWaterFound, cv2.MORPH_OPEN, kernel)
    cv2.imshow("mask created",isWaterFound)
    cv2.imwrite(outputFolder + vidpath[-12:-4] +'newMask_direct.png', isWaterFound)
    cv2.imshow("old mask", trueMask)
    cv2.waitKey(0)
    FigureOutNumbers(isWaterFound, trueMask)
    completeVid = Imagetransformations.importandgrayscale(vidpath,numFrames,dFactor)
    maskedImg = maskFrameWithMyMask(completeVid[minrand:height-minrand, minrand:width-minrand,int(numFrames/2)],isWaterFound)
    cv2.imwrite(outputFolder + vidpath[-12:-4] + 'Masked_frame_from_video.png', maskedImg)


def FigureOutNumbers(createdMask, trueMask):
    print("percent accuracy: " + str(100*np.sum(trueMask == createdMask) / createdMask.size))
    cond1 = (createdMask != trueMask) & (trueMask == 0)
    falsePos =  createdMask[cond1]
    print("percent false positive: " + str(100*(len(falsePos)/trueMask.size)))

    cond2 = (createdMask != trueMask) & (trueMask == 255)
    falseNeg = createdMask[cond2]
    print("percent false negative: "+ str(100*(len(falseNeg)/trueMask.size)))

def maskFrameWithMyMask(frameFromVid,ourMask):
    frameFromVid[ourMask == 0] = 0
    cv2.imshow("windowName", frameFromVid)
    cv2.waitKey(0)
    return frameFromVid

def regularizeFrame(myMask, probabilityMask,gamma):
    newMask = np.zeros(myMask.shape)
    for i in range(1,myMask.shape[0]-1):
        for j in range(1,myMask.shape[1]-1):
            zeros = probabilityMask[i,j] + (gamma * regularizeHelper(myMask,i,j,0))
            twoFiftyFive = 1 - probabilityMask[i,j] + (gamma * regularizeHelper(myMask,i,j,255))
            if zeros < twoFiftyFive:
                newMask[i,j] = 0
            else:
                newMask[i, j] = 255
    return newMask
def regularizeHelper(myMask, i, j,checkValue):
    up = checkValue != myMask[i+1,j]
    down =checkValue != myMask[i - 1, j]
    left = checkValue != myMask[i, j-1]
    right = checkValue != myMask[i, j + 1]
    topleft = checkValue != myMask[i + 1, j -1]
    topRight = checkValue != myMask[i + 1, j + 1]
    bottomLeft = checkValue != myMask[i - 1, j - 1]
    bottomRight = checkValue != myMask[i - 1, j + 1]
    sum1 = int(up) + int(down) + int(left) +int(right) +int(topleft) + int(topRight) + int(bottomLeft) + int(bottomRight)
    return sum1





# def testVid(vidpath,maskpath, numFrames, dFactor, densityMode,boxSize,numbofFrameSearch,numbofSamples,patchSize,numFramesAvg):
#     feature, isWater = ct.moduleB(vidpath,maskpath,numFrames,dFactor,densityMode,boxSize,numbofFrameSearch,numbofSamples,patchSize,numFramesAvg)
#     isWater = isWater.reshape(isWater.shape[1])
#
#     # load the SVM model
#     model = cv2.ml.SVM_load(".//model_4.xml")
#     isWaterFound = np.zeros((numbofSamples),  dtype = np.int)
#     for i in range(feature.shape[0]):
#         h = feature[i,:]
#         h = h.reshape(-1, h.shape[0])
#         label = model.predict(h)
#         if(label[1]==1):
#             isWaterFound[i] = True
#         else:
#             isWaterFound[i] = False
#
#     #print(isWater==isWaterFound)
#     print(np.sum(isWater==isWaterFound))