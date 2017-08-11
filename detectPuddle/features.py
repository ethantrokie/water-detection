import cv2
import numpy as np
from skimage import feature as feat
import math
import plotFuncs

#functions that find features

def fourierTransform(completeVid,randx,randy,randz,boxSize,temporalLength):
    counter = 0
    addTo =math.floor(boxSize/2)
    # sets arrays that will be filled in with mode info
    subBox = completeVid[randy-addTo:randy+addTo,randx-addTo:randx+addTo,randz-int(temporalLength/2):int(randz+temporalLength/2)]
    arrforFourier = np.zeros((1,temporalLength),dtype=np.float32)
    kernel = np.ones((boxSize, boxSize), np.float32) / (boxSize * boxSize)

    while (counter < temporalLength):
        frame = subBox[:,:,counter] * kernel
        total = np.sum(frame)
        arrforFourier[0,counter] = total
        counter += 1

    fourier = np.fft.fft(arrforFourier)
    amplitude_spectrum = np.abs(fourier)
    sumofSignal = np.sum(amplitude_spectrum)
    temporalFeature = amplitude_spectrum/sumofSignal
    temporalFeature = temporalFeature.astype(np.float32)
    return temporalFeature

def SpatialFeatures(completeVid,randx,randy,randz,patchSize,NumFramesAvg):
    addTo = math.floor(patchSize/2)
    subBox = completeVid[randy - addTo:randy + addTo, randx - addTo:randx + addTo, randz - int(NumFramesAvg / 2):int(randz + NumFramesAvg / 2)]
    # compute the Local Binary Pattern representation of the image, and then use the LBP representation to build the histogram of patterns
    lbp = np.zeros(subBox.shape)
    for i in range(NumFramesAvg):
        lbp[:,:,i] = feat.local_binary_pattern(subBox[:,:,i],8, 1)

    # return the histogram of Local Binary Patterns
    histtemp = np.reshape(lbp, (np.product(lbp.shape[1] * lbp.shape[0]*lbp.shape[2])))
    hist, bins = np.histogram(histtemp, bins=np.arange(0, 257), range=(0, 255), normed=True)
    hist = hist.astype(np.float32)
    return hist

def fourierTransformFullImage(completeVid,boxSize):
    # sets up variables to open and save video
    numFrames = completeVid.shape[2]
    counter = 0
    # sets array that will be filled in with mode info
    convolveArr = np.zeros(completeVid.shape, dtype=np.float32)
    kernel = np.ones((boxSize, boxSize), np.float32) / (boxSize *boxSize)
    kernel2 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    while (counter < numFrames):
        frame = completeVid[:, :, counter]
        dst = cv2.filter2D(frame, -1, kernel)
        dst = cv2.filter2D(dst, -1, kernel2)

        convolveArr[:, :, counter] = dst
        counter += 1

    fourier = np.fft.fft(convolveArr, axis=2)
    # get L1 norm of fourier
    amplitude_spectrum = np.abs(fourier)
    sumofSignal = np.sum(amplitude_spectrum, 2)
    temporalFeature = np.zeros(amplitude_spectrum.shape)
    for i in range(fourier.shape[2]):
        temporalFeature[:, :, i] = amplitude_spectrum[:, :, i] / sumofSignal
    #plotFuncs.plotTimeandFourier(temporalFeature,convolveArr,30,150,30,30)
    temporalFeature[np.isnan(temporalFeature)] = 255
    temporalFeature[np.isinf(temporalFeature)] = 255
    return temporalFeature

def SpatialFeaturesFullImage(completeVid,patchSize,AverageFrameNum):
    width = completeVid.shape[1]
    height = completeVid.shape[0]
    numFrames = completeVid.shape[2]
    # compute the Local Binary Pattern representation
    # of the image, and then use the LBP representation
    # to build the histogram of patterns
    lbp = np.zeros(completeVid.shape)
    for i in range(numFrames):
        lbp[:,:,i] = feat.local_binary_pattern(completeVid[:,:,i],8, 1)
        #print(i)

    # return the histogram of Local Binary Patterns
    minrval = int(patchSize / 2)
    addTo = math.floor(patchSize / 2)
    hist = np.zeros((height,width,256))
    for i in range(minrval,width - minrval):
        for j in range(minrval, height - minrval):
            subBox = lbp[j - addTo:j + addTo, i - addTo:i + addTo, int(numFrames / 2) - int(AverageFrameNum / 2):int(int(numFrames / 2) + AverageFrameNum / 2)]
            hist[j,i,:], bins = np.histogram(np.ravel(subBox), bins=np.arange(0, 257), range=(0, 255), normed=True)
        #print(i)
    hist = hist.astype(np.float32)
    return hist

def createMask(maskpath,dscale):
    img = cv2.imread(maskpath)
    width = img.shape[1]
    height = img.shape[0]
    img = cv2.resize(img, (int(width / dscale), int(height / dscale)), interpolation=cv2.INTER_CUBIC)
    big = img >= 127
    return big
