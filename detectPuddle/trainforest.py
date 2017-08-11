import cv2
import numpy as np
import containerFunctions as ct
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

def main():
    samples, y_labels = ct.LoopsThroughAllVids("/data/beehive/VideoWaterDatabase/videos/water/",
                                               "/data/beehive/VideoWaterDatabase/masks/water/",
                                               "/data/beehive/VideoWaterDatabase/videos/non_water/", 400  , 2, 0, 4, 200,
                                               3500, 10, 74)
    # np.save("saved_samples_deriv200.npy", samples)
    # np.save("saved_ylabels_deriv200.npy", y_labels)
    # samples = np.load("saved_samples_deriv200.npy")
    # y_labels = np.load("saved_ylabels_deriv200.npy")
    samples = samples.astype(np.float32)
    y_labels = y_labels.astype(int)
    y_labels = y_labels[:,0]
    print(samples.shape)
    print(y_labels.shape)
    model = RandomForestClassifier(n_estimators=40,n_jobs=-1,class_weight="balanced",max_features="log2")
    model.fit(samples,y_labels)
    joblib.dump(model,'tree.pkl')
if __name__ == '__main__':
   main()


