import os

import cv2
import numpy as np


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
pathtoImgs = "/home/etrokie/Downloads/folder for images of water on pavement/"
for img in os.listdir(pathtoImgs):
    pic = cv2.imread(pathtoImgs+img)
    pic = cv2.cvtColor(pic, cv2.COLOR_BGR2HSV)
    pic = pic.astype(float)
    pic = pic[:,:,0] / pic[:,:,1]
    pic = pic.astype(np.uint8)
    cv2.imshow(img,pic)
    cv2.waitKey(0)
    cv2.imwrite("/home/etrokie/Downloads/results/"+img,pic)