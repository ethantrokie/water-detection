import numpy as np
import cv2
import containerFunctions as ct


class Learn:
    def __init__(self, X, y):
        self.est = cv2.ml.SVM_create()
        params = dict(kernel_type=cv2.ml.SVM_RBF, svm_type=cv2.ml.SVM_C_SVC)
        self.est.trainauto(X, y, None, None, params, 4, balanced=True) #kfold=3 (default: 10)


def main():
    # samples, y_labels = ct.LoopsThroughAllVids("/data/beehive/VideoWaterDatabase/videos/water/",
    #                                            "/data/beehive/VideoWaterDatabase/masks/water/",
    #                                            "/data/beehive/VideoWaterDatabase/videos/non_water/", 120, 2, 0, 4,
    #                                            50, 300, 10, 20)
    samples, y_labels = ct.JustOneFolder("/data/beehive/VideoWaterDatabase/videos/water/pond/",
                                         "/data/beehive/VideoWaterDatabase/masks/water/pond/", 170, 2, 0, 4, 50, 300,
                                         10, 20)
    model = Learn(samples,y_labels)
    model.save("./autotrain.xml")

if __name__ == '__main__':
   main()


