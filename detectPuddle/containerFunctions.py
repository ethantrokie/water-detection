import cv2
import numpy as np
import Imagetransformations
import helperFunc
import features
import random
import os
import predictFunctions

def preprocessVideo(path,numFrames, dFactor, densityMode):
    #sets vars if not put in
    cap = cv2.VideoCapture(path)
    if(numFrames is None or numFrames == -1):
        numFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if ((dFactor is None) or (dFactor == -1)):
        dFactor = 1;
    if (densityMode is None):
        densityMode = 0
    gray = Imagetransformations.importandgrayscale(path,numFrames,dFactor)
    #helperFunc.playVid(gray,"grayVid.avi")
    if(densityMode == 1):
        modeFrame = Imagetransformations.getDensitytModeFrame(gray)
        #cv2.imwrite('mode_density.png', modeFrame)
    else:
        modeFrame = Imagetransformations.getDirectModeFrame(gray)
        #cv2.imwrite('mode_Direct.png', modeFrame)
    cap.release()
    residual = Imagetransformations.createResidual(gray,modeFrame)
    return residual

def getFeatures(preprocessedVid,maskpath,dscale,boxSize,TemporalLength,numbofSamples,patchSize,numFramesAvg):
    #sets up vars
    width = preprocessedVid.shape[1]
    height = preprocessedVid.shape[0]
    numFrames = preprocessedVid.shape[2]
    random.seed(0)

    #water == True in mask
    if maskpath == 0:
        mask = np.zeros((height,width))
    else:
        mask = features.createMask(maskpath,dscale)
        mask = mask[:,:,0]

    #sets up arrays for loop
    isWater = np.zeros((1,numbofSamples),dtype=bool)
    temporalArr = np.zeros((numbofSamples, TemporalLength))
    spatialArr = np.zeros((numbofSamples, 256))
    minrand = max(int(boxSize/2+1),int(patchSize/2))
    totalFeatures= np.zeros((numbofSamples,TemporalLength+256))

    #gets array with amount of features
    for i in range(numbofSamples):
        randx = random.randrange(minrand, width - minrand)
        randy = random.randrange(minrand, height - minrand)
        randz = random.randrange(int(TemporalLength/2), numFrames - int(TemporalLength/2))
        isWater[0,i] = mask[randy,randx]
        temporalArr[i,:] = features.fourierTransform(preprocessedVid,randx,randy,randz,boxSize,TemporalLength)
        temporalFeat = features.fourierTransform(preprocessedVid, randx, randy, randz, boxSize, TemporalLength)

        spatialArr[i,:] = features.SpatialFeatures(preprocessedVid,randx,randy,randz,patchSize,numFramesAvg)
        spaceFeat = features.SpatialFeatures(preprocessedVid,randx,randy,randz,patchSize,numFramesAvg)
        combinedFeature = np.concatenate((np.reshape(temporalFeat,(temporalFeat.size)),spaceFeat))
        totalFeatures[i,:] = combinedFeature
    #features.PlotTemporalFeatures(temporalArr, isWater,numbofSamples)
    #features.plotSpatialFeatures(spatialArr,isWater,numbofSamples)

    print("finished computing unified featureSpace")
    isWater = isWater.astype(int)
    return totalFeatures, isWater


def moduleB(vidpath,maskpath, numFrames, dFactor, densityMode,boxSize,NumbofFrameSearch,numbofSamples,patchSize,numFramesAvg):
    preprocess = preprocessVideo(vidpath,numFrames,dFactor,densityMode)
    features, isWater = getFeatures(preprocess,maskpath,dFactor,boxSize,NumbofFrameSearch,numbofSamples,patchSize,numFramesAvg)
    return features, isWater

#for training the classifier
def LoopsThroughAllVids(pathToVidsFolder,pathToMasksPondFolder,pathToOtherTextures,numFrames, dFactor, densityMode,boxSize,TemporalLength,numbofSamples,patchSize,numFramesAvg):

    #sets up vars
    totalFeatureSet = None
    isWateragg = None
    halfamountofVidsinFolder = 25
    #goes through first 20 vids
    for folders in os.listdir(pathToVidsFolder):
        counter = 0
        for vids in os.listdir(pathToVidsFolder + "/" + folders):
            if counter > halfamountofVidsinFolder:
                break
            nameMask = pathToMasksPondFolder + folders + "/" + vids[:-3] + 'png'
            nameVid = pathToVidsFolder + folders + "/" + vids
            #obtains features for video then concats them to matrix
            feature,isWater = moduleB(nameVid,nameMask,numFrames, dFactor, densityMode,boxSize,TemporalLength,numbofSamples,patchSize,numFramesAvg)
            if totalFeatureSet is None:
                totalFeatureSet = feature
            else:
                totalFeatureSet = np.concatenate((totalFeatureSet,feature),axis=0)
            if isWateragg is None:
                isWateragg = isWater
            else:
                isWateragg = np.concatenate((isWateragg,isWater),axis=1)
            counter+= 1
            print(counter)
        print(folders)
    halfamountofVidsinFolder = 15
    for folders in os.listdir(pathToOtherTextures):
        counter = 0
        for vids in os.listdir(pathToOtherTextures + folders):
            if counter > halfamountofVidsinFolder:
                break
            nameVid = pathToOtherTextures + folders + "/" + vids
            # obtains features for video then concats them to matrix
            feature, isWater = moduleB(nameVid, 0, numFrames, dFactor, densityMode, boxSize, TemporalLength,
                                       numbofSamples, patchSize, numFramesAvg)
            if totalFeatureSet is None:
                totalFeatureSet = feature
            else:
                totalFeatureSet = np.concatenate((totalFeatureSet, feature), axis=0)
            if isWateragg is None:
                isWateragg = isWater
            else:
                isWateragg = np.concatenate((isWateragg, isWater), axis=1)
            counter += 1
            print(counter)

    #turns water into correct orientation
    isWateragg = isWateragg * 1
    isWateragg = np.transpose(isWateragg)
    return totalFeatureSet, isWateragg

def JustOneFolder(pathToVidsFolder,pathToMasksPondFolder,numFrames, dFactor, densityMode,boxSize,TemporalLength,numbofSamples,patchSize,numFramesAvg):

    #sets up vars
    totalFeatureSet = None
    isWateragg = None
    halfamountofVidsinFolder = 1
    #goes through first 20 vids
    counter = 0
    for vids in os.listdir(pathToVidsFolder):
        if counter > halfamountofVidsinFolder:
            break
        nameMask = pathToMasksPondFolder + vids[:-3] + 'png'
        nameVid = pathToVidsFolder + vids
        #obtains features for video then concats them to matrix
        feature,isWater = moduleB(nameVid,nameMask,numFrames, dFactor, densityMode,boxSize,TemporalLength,numbofSamples,patchSize,numFramesAvg)
        if totalFeatureSet is None:
            totalFeatureSet = feature
        else:
            totalFeatureSet = np.concatenate((totalFeatureSet,feature),axis=0)
        if isWateragg is None:
            isWateragg = isWater
        else:
            isWateragg = np.concatenate((isWateragg,isWater),axis=1)
        counter+= 1
        print(counter)
    #turns water into correct orientation
    isWateragg = isWateragg * 1
    isWateragg = np.transpose(isWateragg)
    return totalFeatureSet, isWateragg
