import cv2
import numpy as np
from math import e, sqrt, pi

#plays videos of type VideoWriter
def playVid(arr, outputName):
    arr = arr.astype(np.uint8)
    width = arr.shape[1]
    height = arr.shape[0]
    fps = 20
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(outputName, fourcc, fps, (width, height), False)
    counter = 0
    numFrames = arr.shape[2]
    while counter < numFrames:
        frame = arr[:, :, counter]
        out.write(frame)
        cv2.imshow('justRunVid', frame)
        if cv2.waitKey(400) & 0xFF == ord('q'):
            break
        counter+=1
    cv2.destroyAllWindows()
    out.release()

def saveVid(arr, outputName):
    arr = arr.astype(np.uint8)
    width = arr.shape[1]
    height = arr.shape[0]
    fps = 20
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(outputName, fourcc, fps, (width, height), False)
    counter = 0
    numFrames = arr.shape[2]
    while counter < numFrames:
        frame = arr[:, :, counter]
        out.write(frame)
        counter+=1
    cv2.destroyAllWindows()
    out.release()
