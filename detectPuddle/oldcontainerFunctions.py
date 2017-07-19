import cv2
import numpy as np
import Imagetransformations
import helperFunc
import features
import random


def preprocessVideo(path,numFrames, dFactor, densityMode):
    cap = cv2.VideoCapture(path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
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
    #helperFunc.playVid(residual, "residual_most_recent.avi")
    return residual

def getFeatures(preprocessedVid,maskpath,dscale,boxSize,NumbofFramesDone,samples):
    width = preprocessedVid.shape[1]
    height = preprocessedVid.shape[0]
    numFrames = preprocessedVid.shape[2]
    random.seed()
    #water == True in mask
    mask = features.createMask(maskpath,dscale)
    mask = mask[:,:,0]
    temporalFeat = features.fourierTransform(preprocessedVid)

    #tempFeatFinal = temporalFeature[mask]
    features.PlotTemporalFeaturesFullImage(temporalFeat)
    #spaceFeat = features.SpatialFeatures()
    #spaceFeatFinal = spaceFeat[np.invert(mask)]
    print("finished computing unified featureSpace")
    #return combinedFeature


def moduleB(vidpath,maskpath, numFrames, dFactor, densityMode,boxSize,NumbofFrameSearch):
    preprocess = preprocessVideo(vidpath,numFrames,dFactor,densityMode)
    FeatureVector = getFeatures(preprocess,maskpath,dFactor,boxSize,NumbofFrameSearch,100)
    return FeatureVector
