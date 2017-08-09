import cv2
import numpy as np

import containerFunctions as ct
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

def main():
    #samples, y_labels = ct.LoopsThroughAllVids("/data/beehive/VideoWaterDatabase/videos/water/","/data/beehive/VideoWaterDatabase/masks/water/","/data/beehive/VideoWaterDatabase/videos/non_water/",500,2,0,4,200,3500,10,74)
    # samples, y_labels = ct.JustOneFolder("/data/beehive/VideoWaterDatabase/videos/water/pond/",
    #                                      "/data/beehive/VideoWaterDatabase/masks/water/pond/", 170, 2, 0, 4, 50, 300,
    #                                      10, 20)

    #samples = np.load("saved_samples_4.npy")
    #y_labels = np.load("saved_ylabels_4.npy")
    # np.save("saved_samples_4.npy",samples)
    # np.save("saved_ylabels_4.npy",y_labels)
    samples, y_labels = ct.LoopsThroughAllVids("/data/beehive/VideoWaterDatabase/videos/water/",
                                               "/data/beehive/VideoWaterDatabase/masks/water/",
                                               "/data/beehive/VideoWaterDatabase/videos/non_water/", 400, 2, 0, 4, 50,
                                               3500, 10, 24)
    # np.save("saved_samples_50frames.npy", samples1)
    # np.save("saved_ylabels_50frames.npy", y_labels1)
    samples = samples.astype(np.float32)
    y_labels = y_labels.astype(int)
    y_labels = y_labels[:,0]

    print(samples.shape)
    print(y_labels.shape)
    model = RandomForestClassifier(n_estimators=50,n_jobs=-1,class_weight="balanced",max_features="log2")
    model.fit(samples,y_labels)    # compute the hog feature
    joblib.dump(model,'tree50frame.pkl')
if __name__ == '__main__':
   main()


