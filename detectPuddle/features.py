import cv2
import numpy as np
from scipy import stats
from math import e, sqrt, pi
import tkinter
import matplotlib.pyplot as plt
from skimage import feature as feat
import helperFunc as helper
import math


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

def fourierTransformFullImage(completeVid,boxSize,mask):
    # sets up variables to open and save video
    numFrames = completeVid.shape[2]
    counter = 0
    # sets array that will be filled in with mode info
    convolveArr = np.zeros(completeVid.shape, dtype=np.float32)
    kernel = np.ones((boxSize, boxSize), np.float32) / (boxSize *boxSize)
    while (counter < numFrames):
        frame = completeVid[:, :, counter]
        dst = cv2.filter2D(frame, -1, kernel)
        convolveArr[:, :, counter] = dst
        counter += 1

    fourier = np.fft.fft(convolveArr, axis=2)
    # get L1 norm of fourier
    amplitude_spectrum = np.abs(fourier)
    sumofSignal = np.sum(amplitude_spectrum, 2)
    temporalFeature = np.zeros(amplitude_spectrum.shape)
    for i in range(fourier.shape[2]):
        temporalFeature[:, :, i] = amplitude_spectrum[:, :, i] / sumofSignal
    # plotTimeandFourier(temporalFeature,convolveArr,30,200,20,26)
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
        print(i)

    # return the histogram of Local Binary Patterns
    minrand = int(patchSize / 2)
    addTo = math.floor(patchSize / 2)
    hist = np.zeros((height,width,256))
    for i in range(minrand,width - minrand):
        for j in range(minrand, height - minrand):
            subBox = lbp[j - addTo:j + addTo, i - addTo:i + addTo, int(numFrames / 2) - int(AverageFrameNum / 2):int(int(numFrames / 2) + AverageFrameNum / 2)]
            hist[j,i,:], bins = np.histogram(np.ravel(subBox), bins=np.arange(0, 257), range=(0, 255), normed=True)
        print(i)
    hist = hist.astype(np.float32)
    return hist


#functions that plot spatial features

def plotSpatialFeatures(histMatrix,isWater,numSamples):
    water = histMatrix[np.reshape(isWater, (numSamples))]
    notwater = histMatrix[np.invert(np.reshape(isWater, (numSamples)))]
    fig, ax = plt.subplots()
    ax.legend(prop={'size': 10})
    ax.set_title('Water vs not Water')
    water1 = np.average(water,0)
    notwater1 = np.average(notwater,0)

    barwidth = 0.35

    ax.bar(np.arange(0,256), water1, barwidth, color='b', label="Water")
    ax.bar(np.arange(0,256) + barwidth, notwater1, barwidth, color='r',label="Not Water")
    plt.legend(loc='upper right')
    plt.show()


def plotTimeandFourier(fourierVid,timeVid,xwater,ywater,xnotwater,ynotwater):
    waterTime1 = timeVid[ywater,xwater,1:]
    waterFourier1 = fourierVid[ywater,xwater,1:]
    notwaterTime1 = timeVid[ynotwater,xnotwater,1:]
    notwaterFourier1 = fourierVid[ynotwater,xnotwater,1:]

    X = range(len(waterTime1-1))
    plt.figure(1)
    plt.subplot(211)
    plt.plot(X, waterTime1, 'b', X, notwaterTime1, 'r')

    plt.subplot(212)
    plt.plot(X, waterFourier1, 'b', X, notwaterFourier1, 'r')
    plt.show()


def PlotTemporalFeatures(temporalsignals,isWater,numSamples):
    fig, ax = plt.subplots()
    water = temporalsignals[np.reshape(isWater,(numSamples))]
    notwater = temporalsignals[np.invert(np.reshape(isWater,(numSamples)))]
    waterx = water[:,1]
    waterx = np.reshape(waterx, (np.product(waterx.shape)))
    watery = water[:,2]
    watery = np.reshape(watery, (np.product(watery.shape)))
    notwaterx = notwater[:,1]
    notwaterx = np.reshape(notwaterx, (np.product(notwaterx.shape)))
    notwatery = notwater[:,2]
    notwatery = np.reshape(notwatery, (np.product(notwatery.shape)))
    ax.set_title('Water vs not Water')
    ax.plot(waterx, watery, 'bo', label="water")
    ax.plot(notwaterx, notwatery, 'ro',label="not water")
    plt.legend(loc='upper right')
    plt.show()



def createMask(maskpath,dscale):
    img = cv2.imread(maskpath)
    width = img.shape[1]
    height = img.shape[0]
    img = cv2.resize(img, (int(height / dscale), int(width / dscale)), interpolation=cv2.INTER_CUBIC)
    big = img >= 127
    return big


# def plotSpatialFeatures(spatialFeat):
#     width = spatialFeat.shape[1]
#     height = spatialFeat.shape[0]
#     numbofFrames = spatialFeat.shape[2]
#     colors = ['blue', 'red']
#     names = ['Water 1', 'not water']
#     fig, ax = plt.subplots()
#     ax.legend(prop={'size': 10})
#     ax.set_title('Water Vs water')
#
#     water = spatialFeat[0:21, 0:10]
#     water2 = spatialFeat[int((height / 2)):int(height / 2 + 21), 0:20]
#     water3 = spatialFeat[int((height / 2)):int(height / 2 + 21), width - 21:width - 1]
#     hist = np.zeros((256,numbofFrames))
#     for i in range(numbofFrames):
#         watertemp = np.reshape(water2[:,:,i], (np.product(water2.shape[1]*water2.shape[0])))
#         hist[:,i], bins = np.histogram(watertemp, bins=np.arange(0, 257), range=(0, 255), normed=True)
#     histWater = np.aveqrage(hist,1)
#
#     notWater = spatialFeat[0:21, width - 11:width - 1]
#     hist2 = np.zeros((256, numbofFrames))
#     for i in range(numbofFrames):
#         notwaterTemp = np.reshape(water3[:,:,i], (np.product(water3.shape[1]*water3.shape[0])))
#         hist2[:, i], bins = np.histogram(notwaterTemp, bins=np.arange(0, 257), range=(0, 255), normed=True)
#     histNotWater = np.average(hist2, 1)
#
#     barwidth = 0.35
#
#     rects1 = ax.bar(np.arange(0,256), histWater, barwidth, color='b',)
#     rects2 = ax.bar(np.arange(0,256) + barwidth, histNotWater, barwidth, color='r')
#
#
#
#
#     plt.show()

# def PlotTemporalFeaturesFullImage(temporalsignal):
#     width = temporalsignal.shape[1]
#     height = temporalsignal.shape[0]
#     #waterx = temporalsignal[height-11:height-1,0:10,1]
#     waterx = temporalsignal[0:11, 0:10, 1]
#     waterx = np.reshape(waterx, (np.product(waterx.shape)))
#     #watery = temporalsignal[height - 11:height - 1, 0:10, 3]
#     watery = temporalsignal[0:11, 0:10, 2]
#     watery = np.reshape(watery, (np.product(watery.shape)))
#     notWaterx = temporalsignal[0:10,width-11:width-1,1]
#     notWaterx = np.reshape(notWaterx, (np.product(notWaterx.shape)))
#     notWatery = temporalsignal[0:10, width - 11:width - 1, 2]
#     notWatery = np.reshape(notWatery, (np.product(notWatery.shape)))
#     plt.plot(waterx,watery, 'bo', notWaterx, notWatery,'ro')
#     plt.show()
