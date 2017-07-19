import cv2
import numpy as np
from scipy import stats
from math import e, sqrt, pi
import helperFunc
#probably going to have to change when put on node,
# maybe write other class to compile frames into video

#make this into a class
def importandgrayscale(path,numbframes,dscale):
    #sets up variables to open and save video
    cap = cv2.VideoCapture(path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #set amount of frames want to look at
    numFrames = numbframes
    counter = 0
    completeVid = np.zeros((int(height/dscale), int(width/dscale), numFrames), dtype=np.uint8)
    #main body
    while(cap.isOpened() & (counter < numFrames)):
        ret, frame = cap.read()
        #checks to makes sure frame is valid
        if ret == True:
            if frame is not None:
                #turns frame gray and saves
                frame = cv2.resize(frame,(int(width/dscale),int(height/dscale)),interpolation=cv2.INTER_CUBIC)
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                completeVid[:, :, counter] = gray
            else:
                break
        else:
            continue
        counter += 1

    cap.release()
    print("finshed Black and white conversion")
    return completeVid

def getDirectModeFrame(completeVid):
    # get mode and saves it - saves it three times so can save correctly
    modeFrame = stats.mode(completeVid, 2)
    modeFrameFinal = (modeFrame[0])[:,:,0]
    print("got direct mode frame")
    cv2.imwrite("mode_frame.png",modeFrameFinal)
    return modeFrameFinal

def getDensitytModeFrame(completeVid):
    #set up to open video and save image
    width = completeVid.shape[1]
    height = completeVid.shape[0]
    numFrames = completeVid.shape[2]

    # get mode and saves it - saves it three times so can save correctly
    M = np.zeros((height, width, 256))
    stdd = np.std(completeVid,2)
    stdd = stdd*3.5/float(numFrames**(1/3))
    for i in range(256):
        temp = np.zeros((height,width))

        for j in range(numFrames):
            placeholder = i - completeVid[:,:,j]
            temp = temp + (1/(sqrt(2 * pi) * stdd) * np.exp(-0.5 * (placeholder[:,:] / stdd) ** 2))
        M[:,:,i] = temp
        print(i)

    mode = np.argmax(M,2);
    print("got density mode frame")
    return mode


def findmin(completeVid):
    minframe = completeVid.min(2)
    return minframe

def createResidual(completeVid,modeImg):
    #sets up variables to open and save video
    minFrame = findmin(completeVid)

    #sets amount of frames you want to look at
    numFrames = completeVid.shape[2]
    counter = 0
    frame_write = np.zeros(completeVid.shape,dtype=np.float32)
    while (counter < numFrames):
        #subracts mode frame from actual frame
        frame = completeVid[:,:,counter]
        frame_write[:,:,counter]= frame.astype(np.float32) - modeImg.astype(np.float32) + 127 # minFrame.astype(np.float32)
        counter += 1
    #releases video containers
    print("got residual video")
    return frame_write


