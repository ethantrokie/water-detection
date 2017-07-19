from sklearn import svm
import numpy as np
import cv2
import containerFunctions as ct
from sklearn.externals import joblib
import predictFunctions

def main():
    samples, y_labels = ct.JustOneFolder("/data/beehive/VideoWaterDatabase/videos/water/pond/","/data/beehive/VideoWaterDatabase/masks/water/pond/",150,2 ,0,4,50,300,10,20)
    samples = samples.astype(np.float64)
    y_labels = y_labels.astype(int)
    y_labels = y_labels[:,0]
    print(samples.shape)
    print(y_labels.shape)
    clf = svm.SVC(C=.01,gamma=.005,kernel='rbf')
    clf.fit(samples, y_labels)# compute the hog feature
    joblib.dump(clf, 'model_linear.pkl')
    predictFunctions.predictFromOtherSVM("/data/beehive/ForTesting/videos/water/pond/pond_024.avi", "/data/beehive/ForTesting/masks/water/pond/pond_024.png",
                                         50, 2, 0, 4, 50, 100, 10, 32,clf)
if __name__ == '__main__':
   main()
