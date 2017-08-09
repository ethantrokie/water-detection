
import numpy as np
import matplotlib.pyplot as plt

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
