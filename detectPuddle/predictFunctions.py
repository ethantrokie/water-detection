import math

import cv2
import numpy as np
import sys
import Imagetransformations
import containerFunctions as ct
import features
import helperFunc
from sklearn.externals import joblib


def moduleE(vidpath,maskpath,outputFolder, numFrames, dFactor, densityMode,boxSize,patchSize,numFramesAvg,numVids):
    maskArr = None
    for i in range(numVids):
        mask,trueMask = testFullVid(vidpath, maskpath, outputFolder, numFrames, dFactor, densityMode, boxSize, patchSize, numFramesAvg,i)
        if mask is None:
            print("didn't have enough frames to run this many times")
            break
        if maskArr is None:
            maskArr = mask
        else:
            maskArr = np.dstack((maskArr,mask))

    if maskArr is None:
        print("video is too short, no mask generated")
        sys.exit()
    finalMask = np.sum(maskArr,2)
    logical = finalMask < (255 * 2 * numVids/4 )
    finalMask[logical] = 0
    finalMask[finalMask != 0] = 255
    cv2.imshow("normalizedMask",finalMask)
    cv2.imwrite(outputFolder +"FinalMask.png",finalMask)
    cv2.waitKey(0)
    if trueMask is not None:
        FigureOutNumbers(finalMask, trueMask)


def moduleD(vidpath,maskpath,outputFolder, numFrames, dFactor, densityMode,boxSize,numbofFrameSearch,numbofSamples,patchSize,numFramesAvg,numVids):
    maskArr = None
    for i in range(numVids):
        mask,trueMask = testFullVid(vidpath, maskpath, outputFolder, numFrames, dFactor, densityMode, boxSize, numbofFrameSearch,
                    numbofSamples, patchSize, numFramesAvg,i)
        if mask is None:
            print("didn't have enough frames to run this many times")
            break
        if maskArr is None:
            maskArr = mask
        else:
            maskArr = np.dstack((maskArr,mask))
    if maskArr is None:
            exit(0)
    finalMask = np.sum(maskArr,2)
    logical = finalMask < (255 * numVids/2)
    finalMask[logical] = 0
    finalMask[finalMask != 0] = 255
    cv2.imshow("normalizedMask",finalMask)
    cv2.imwrite("FinalMask.png",finalMask)
    cv2.waitKey(0)
    if trueMask is not None:
        FigureOutNumbers(finalMask, trueMask)

def moduleC(vidpath,maskpath,outputFolder, numFrames, dFactor, densityMode,boxSize,patchSize,numFramesAvg,vidNum):
    preprocess = ct.preprocessVideo(vidpath,numFrames,dFactor,densityMode,vidNum)
    if preprocess is None:
        return None,None
    helperFunc.saveVid(preprocess, outputFolder + vidpath[-12:-4] + "residual_most_recent.avi")
    features, isWater = getFeatures(preprocess,maskpath,dFactor,boxSize,patchSize,numFramesAvg)
    return features, isWater
def getFeatures(preprocessedVid,maskpath,dscale,boxSize,patchSize,numFramesAvg):
    #sets up vars
    width = preprocessedVid.shape[1]
    height = preprocessedVid.shape[0]

    #water == True in mask
    if maskpath is not 0:
        mask = features.createMask(maskpath,dscale)
        mask = mask[:,:,0]
        isWater = mask.astype(np.uint8)
    else:
        isWater = None

    #size of edge around the image-- will later crop this part out
    minrand = max(int(boxSize / 2 + 1), int(patchSize / 2))

    #obtain temporal feature and corp the image
    temporalFeat = features.fourierTransformFullImage(preprocessedVid,boxSize)
    temporalFeat[0:minrand,:] = 0
    temporalFeat[:,0:minrand] = 0
    temporalFeat[height-minrand:height, :] = 0
    temporalFeat[:,width-minrand:width] = 0
    spaceFeat = features.SpatialFeaturesFullImage(preprocessedVid,patchSize,numFramesAvg)
    combinedFeature = np.concatenate((temporalFeat,spaceFeat),2)


    print("finished computing unified featureSpace")

    combinedFeature = combinedFeature.astype(np.float32)
    return combinedFeature, isWater

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

def testFullVid(vidpath,maskpath,outputFolder, numFrames, dFactor, densityMode,boxSize,patchSize,numFramesAvg,vidNum):
    feature, trueMask = moduleC(vidpath,maskpath,outputFolder,numFrames,dFactor,densityMode,boxSize,patchSize,numFramesAvg,vidNum)
    if feature is None:
        return None,None
    width = feature.shape[1]
    height = feature.shape[0]
    minrand = max(int(boxSize / 2 + 1), int(patchSize / 2))
    # load the SVM model
    model = joblib.load('tree150log2.pkl')
    isWaterFound = np.zeros((height,width),  dtype = np.int)
    newShape = feature.reshape((feature.shape[0] * feature.shape[1], feature.shape[2]))
    prob = model.predict_proba(newShape)[:, 0]
    prob = prob.reshape((feature.shape[0], feature.shape[1]))
    probabilityMask = prob
    isWaterFound[prob<.5] = True
    isWaterFound[prob>.5] = False
    isWaterFound = isWaterFound.astype(np.uint8)
    isWaterFound = isWaterFound[minrand:height-minrand, minrand:width-minrand]
    if trueMask is not None:
        trueMask = trueMask[minrand:height-minrand,minrand:width-minrand]
        trueMask[trueMask == 1] = 255
    isWaterFound[isWaterFound == 1] = 255
    beforeReg = outputFolder + str(vidNum) + '_before_regularization' + '.png'
    cv2.imwrite(beforeReg, isWaterFound)
    probabilityMask = probabilityMask[minrand:height-minrand, minrand:width-minrand]
    #probabilityMask = (probabilityMask-np.min(probabilityMask))/(np.max(probabilityMask)-np.min(probabilityMask))
    for i in range(11):
        isWaterFound = regularizeFrame(isWaterFound,probabilityMask,.2)
    #isWaterFound = cv2.morphologyEx(isWaterFound, cv2.MORPH_OPEN, kernel)
    width = isWaterFound.shape[1]
    height = isWaterFound.shape[0]
    isWaterFound = isWaterFound[11:height-11, 11:width-11]
    trueMask = trueMask[11:height-11, 11:width-11]
    if trueMask is not None:
        cv2.imshow("mask created",isWaterFound)
    cv2.imwrite(outputFolder + str(vidNum) +'newMask_direct.png', isWaterFound)
    if trueMask is not None:
        cv2.imshow("old mask", trueMask)
        cv2.waitKey(0)
    if trueMask is not None:
        FigureOutNumbers(isWaterFound, trueMask)
    completeVid = Imagetransformations.importandgrayscale(vidpath,numFrames,dFactor,vidNum)
    maskedImg = maskFrameWithMyMask(completeVid[11:height-11, 11:width-11,int(numFrames/2)],isWaterFound)
    cv2.imwrite(outputFolder + str(vidNum) + 'Masked_frame_from_video.png', maskedImg)
    return isWaterFound, trueMask



# def testFullVid(vidpath,maskpath,outputFolder, numFrames, dFactor, densityMode,boxSize,numbofFrameSearch,numbofSamples,patchSize,numFramesAvg,vidNum):
#     feature, trueMask = moduleC(vidpath,maskpath,outputFolder,numFrames,dFactor,densityMode,boxSize,numbofFrameSearch,numbofSamples,patchSize,numFramesAvg,vidNum)
#     if feature is None:
#         return None, None
#     width = feature.shape[1]
#     height = feature.shape[0]
#     minrand = max(int(boxSize / 2 + 1), int(patchSize / 2))
#     # load the SVM model
#     model = cv2.ml.SVM_load("with_motion_4500_samples_rbf.xml")
#     isWaterFound = np.zeros((height,width),  dtype = np.int)
#     probabilityMask = np.zeros((height,width),  dtype = np.float64)
#     for i in range(height):
#         for j in range(width):
#             h = feature[i,j,:]
#             h = h.reshape(-1, h.shape[0])
#             dist = model.predict(h,0,1)[1]
#             prob = 1/(1+math.exp(-1*dist))
#             probabilityMask[i, j] = 1 - prob
#             if(prob < 0.5):
#                 isWaterFound[i,j] = True
#             else:
#                 isWaterFound[i,j] = False
#         #print(i)
#
#     isWaterFound = isWaterFound.astype(np.uint8)
#     isWaterFound = isWaterFound[minrand:height-minrand, minrand:width-minrand]
#     if trueMask is not None:
#         trueMask = trueMask[minrand:height-minrand,minrand:width-minrand]
#         trueMask[trueMask == 1] = 255
#     isWaterFound[isWaterFound == 1] = 255
#     beforeReg = outputFolder + str(vidNum) + '_before_regularization' + '.png'
#     cv2.imwrite(beforeReg, isWaterFound)
#     probabilityMask = probabilityMask[minrand:height-minrand, minrand:width-minrand]
#     probabilityMask = (probabilityMask-np.min(probabilityMask))/(np.max(probabilityMask)-np.min(probabilityMask))
#     for i in range(11):
#         isWaterFound = regularizeFrame(isWaterFound,probabilityMask,.2)
#     #isWaterFound = cv2.morphologyEx(isWaterFound, cv2.MORPH_OPEN, kernel)
#     if trueMask is not None:
#         cv2.imshow("mask created",isWaterFound)
#     cv2.imwrite(outputFolder + str(vidNum) +'newMask_direct.png', isWaterFound)
#     if trueMask is not None:
#         cv2.imshow("old mask", trueMask)
#         cv2.waitKey(0)
#     if trueMask is not None:
#         FigureOutNumbers(isWaterFound, trueMask)
#     completeVid = Imagetransformations.importandgrayscale(vidpath,numFrames,dFactor,vidNum)
#     maskedImg = maskFrameWithMyMask(completeVid[minrand:height-minrand, minrand:width-minrand,int(numFrames/2)],isWaterFound)
#     cv2.imwrite(outputFolder + str(vidNum) + 'Masked_frame_from_video.png', maskedImg)
#     return isWaterFound, trueMask


















