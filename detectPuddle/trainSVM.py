import numpy as np
import cv2
import containerFunctions as ct


class StatModel(object):
   def load(self, fn):
       self.model.load(fn)
   def save(self, fn):
       self.model.save(fn)
class SVM(StatModel):
    def __init__(self, C = 0.01, gamma = .05):
       self.model = cv2.ml.SVM_create()
       self.model.setGamma(gamma)
       self.model.setC(C)
       self.model.setKernel(cv2.ml.SVM_LINEAR)
       self.model.setType(cv2.ml.SVM_C_SVC)
    def train(self, samples, responses):
       self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)
    def predict(self, samples):
        return self.model.predict(samples,1)
        #return self.model.predict(samples,1)[1].ravel()
def main():
    samples, y_labels = ct.LoopsThroughAllVids("/data/beehive/VideoWaterDatabase/videos/water/","/data/beehive/VideoWaterDatabase/masks/water/","/data/beehive/VideoWaterDatabase/videos/non_water/",370,2,0,4,200,4000,10,20)
    # samples, y_labels = ct.JustOneFolder("/data/beehive/VideoWaterDatabase/videos/water/pond/",
    #                                      "/data/beehive/VideoWaterDatabase/masks/water/pond/", 170, 2, 0, 4, 50, 300,
    #                                      10, 20)
    samples = samples.astype(np.float32)
    y_labels = y_labels.astype(int)
    y_labels = y_labels[:,0]
    print(samples.shape)
    print(y_labels.shape)
    model = SVM()
    model.train(samples, y_labels)    # compute the hog feature
    model.save("./with_motion_2000_samples_linear.xml")
if __name__ == '__main__':
   main()


