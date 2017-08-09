import matplotlib.pyplot as plt
from sklearn import svm
import numpy as np


samples = np.load("saved_samples_more.npy")
y_labels = np.load("saved_ylables_more.npy")
clf = svm.SVC(C=.1,cache_size=1200, class_weight="balanced")

clf.fit(samples,y_labels)
clf.predict()
plt.show()