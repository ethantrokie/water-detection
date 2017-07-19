import cv2
import numpy as np

cap = cv2.VideoCapture("canal_001.avi")

fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'XVID')

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
dscale = 1

out = cv2.VideoWriter("code_debugging_video.avi", fourcc, fps, (int(width / dscale), int(height / dscale)), False)

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        frame = cv2.resize(frame,(int(width/dscale),int(height/dscale)))
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        out.write(gray)
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("finshed Black and white conversion")


cap2 = cv2.VideoCapture("code_debugging_video.avi")
out2 = cv2.VideoWriter("code_debugging_video_2.avi", fourcc, fps, (int(width / dscale), int(height / dscale)), False)

while(cap2.isOpened()):
    ret, frame = cap2.read()
    if ret == True:
        frame = cv2.resize(frame,(int(width/dscale),int(height/dscale)))
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        out.write(gray)
    else:
        break

cap2.release()
out2.release()
cv2.destroyAllWindows()
print("read write second time")


cap3 = cv2.VideoCapture("code_debugging_video_2.avi")
while(cap3.isOpened()):
    ret, frame = cap3.read()
    if ret == True:
        frame = cv2.resize(frame,(int(width/dscale),int(height/dscale)))
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    else:
        break

cap3.release()
cv2.destroyAllWindows()
print("read write third time")