import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



def importandHSB(path,numFrames,dscale):
    # sets up variables to open and save video
    cap = cv2.VideoCapture(path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # set amount of frames want to look at
    counter = 0
    completeVid = np.zeros((int(height), int(width), numFrames), dtype=np.uint8)
    # main body
    while (cap.isOpened() & (counter < numFrames)):
        ret, frame = cap.read()
        # checks to makes sure frame is valid
        if ret == True:
            if frame is not None:
                #frame = cv2.resize(frame, (int(width / dscale), int(height / dscale)), interpolation=cv2.INTER_CUBIC)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                edged = cv2.Canny(gray, 35, 125)
                cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
                c = max(cnts, key=cv2.contourArea)
                cv2.imshow("first",c)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                counter += 1
            else:
                break
        else:
            continue
        counter += 1

    cap.release()
    print("finshed Black and white conversion")
    return completeVid

#importandHSB("/data/Video_1.MOV",300,1)
pathtoImgs = "/Volumes/ETHAN'S USB/folder for images of water on pavement/"
pathtoMasks= "/Volumes/ETHAN'S USB/result masks/"
ArrpicnotWaterSat = np.zeros((4,100))
ArrpicnotWaterSat = np.zeros((4,100))
ArrpicnotWaterSat = np.zeros((4,100))
ArrpicnotWaterSat = np.zeros((4,100))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for img in os.listdir(pathtoImgs):
    pic = cv2.imread(pathtoImgs+img)
    mask = cv2.imread(pathtoMasks + img)[:,:,0]
    pic = cv2.resize(pic, (int(pic.shape[0] / 15), int(pic.shape[1] / 15)), interpolation=cv2.INTER_CUBIC)
    mask = cv2.resize(mask, (int(mask.shape[0] / 15), int(mask.shape[1] / 15)), interpolation=cv2.INTER_CUBIC)
    # plt.imshow(mask)
    # plt.show()
    mask[mask > 127] = 255
    mask[mask < 127] = 0
    pic = cv2.cvtColor(pic, cv2.COLOR_BGR2HSV_FULL)
    pic = pic.astype(float)


    picHue = pic[:,:,0]
    picSat = pic[:,:,1]
    picVal = pic[:,:,2]


    picwaterHue = picHue[mask == 255 ]
    picnotWaterHue = picHue[mask == 0]
    picwaterSat = picSat[mask == 255]
    picnotWaterSat = picSat[mask == 0]
    picwaterVal = picVal[mask == 255]
    picnotWaterVal = picVal[mask == 0]



    #ax.scatter(picnotWaterHue,picnotWaterSat,picnotWaterVal, c='r')
    ax.scatter(picwaterHue, picwaterSat, picwaterVal, c='b')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')



    #plt.plot( picnotWaterSat[0:100], picnotWaterVal[0:100],'bo')
    #plt.plot(picwaterSat[0:100], picwaterVal[0:100],'ro')
    # plt.set_xlabel('X Label')
    # plt.set_ylabel('Y Label')
plt.show()

cv2.imwrite("/Volumes/ETHAN'S USB/results/" + img, pic)
