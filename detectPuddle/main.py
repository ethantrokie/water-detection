import cv2
import numpy as np
import Imagetransformations
import helperFunc
import features
import containerFunctions as ct
import trainSVM

#ct.LoopsThroughAllVids("/data/beehive/VideoWaterDatabase/videos/water/pond/","/data/beehive/VideoWaterDatabase/masks/water/pond/",100,4,0,4,50,100,20,30,5)
#moduleB(vidpath,maskpath, numFrames, dFactor, densityMode,MUST BE EVEN boxSize,NumbofFrameSearch,numbofSamples,patchSize,numFramesAvg):
#featureMatix = ct.moduleB("pond_016.avi","pond_016_mask.png",100,4,0,4,50,100,20,30)
trainSVM.main()
#ct.moduleB()


