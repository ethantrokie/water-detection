# Water Detection in City Streets Using a Single Camera
## By: Ethan Trokie, Zeeshan Nadir, Nicola Ferrier, Pete Beckman

## Introduction

The goal of this program is to be able to detect flooding in city streets (or elsewhere) and be able to discern between water and non-water regions. This work was primarily based on a paper called “Water Detection through Spatio-Temporal Invariant Descriptors” by Pascal Mettes, Robby T. Tan, Remco C. Veltkamp. The code is mainly broken up into three separate pieces. 

1.	Preprocessing
2.	Feature Extraction
3.	Machine Learning algorithm

Using these three steps they are further broken up into code that trains the classifier and code that classifies the images. In the next sections I will explain how the code works and is formatted.

Dependencies  
•	Numpy  
•	OpenCV  
•	Scipy  
•	Scikit-learn  

# Classifying Video as water/non-water

## Preprocessing

The first step in the process of classifying the videos or training the classifier is processing the video so that they are fit for features extraction (Features are traits of the video that make it distinguishable from other videos). The whole preprocessing procedure is run by a function called preprocessVideo() in containerFunctions.py This function takes in the path to the video, the number of frames that you’re going to look at, the downscaling factor, whether its density mode or not (0 or 1) and vidnumber, which will be explained later. Within this functions it calls subfunctions that can be found in Imagetransformations.py. 

First importandgrayscale() runs, which imports the video and turns it grayscale and returns an np array where each pixel value is stored in its corresponding position in the matrix where the format is (height,width,frame), so (0,0,0) holds the top left pixel in the first frame, then (0,0,1) holds the top left pixel in the second frame. 

Next a “mode frame” is found. A “mode frame” is essentially a theoretical frame where each pixel value is the most common value for that pixel over the whole video. There are two methods of calculating the mode frame, direct method or density method. Density method is slightly better for creating less noisy boundaries, but density method is significantly slower and doesn’t seem to be significantly better than direct mode frame. Direct mode frame calculates the mode frame by just taking the mode of pixel value over time, while the density mode frame is calculated using a kernel density estimation, which creates less noisy results. 
Lastly in this process, the residual video is created. This is created by subtracting the mode frame pixel values from the black and white videos pixel values, so only motion should be seen in the residual video, because anything that is not moving should be removed by the mode frame subtraction.

## Feature Extraction

Feature extraction of the video is what is used as the inputs for the machine learning algorithm. The two main features that will be extracted are going to be a “temporal feature” and a “spatial feature”. The temporal feature is essentially looking at a single pixel throughout the whole video and using the pixel intensity values over time to create a “signal”. Then you take that signal and take the Fourier transformation of that, and transfer it to the frequency where the “signals” are very different between water and non-water. Below in figure 1, the top graph is the pixel values over 200 frames of the video, where the blue is a water pixel, and red is a non-water pixel values. On the top graph the two signals are fairly different are, but they are even more different in the frequency domain, and it would be easier for the classifier to distinguish between different types of motion as well in the frequency domain. 






Time Domain





Frequency Domain
Figure 1

In addition to obtaining the temporal feature, the code obtains a spatial feature. This spatial feature is a Local Binary Pattern of a local patch around each pixel. In the figures below you can see how a LPB can be calculated. First you look at the center pixel and if the values of the pixels surrounding them is larger it gets a zero and if its less than the center pixel is one (figure 2). Then since there are 8 pixels surrounding the center pixel, that can be mapped to an 8 bit number, seen in the figure 3, and that number is the value in the location of the center pixel (figure 4).

Figure 2

figure 3

fig 4

After LBP is taken of a small patch of the image, say 5x5x30, a histogram is created of the values in that square, over 30 frames and that should give a histogram that describes the local area around that pixel. These two features can then be put into the classifier and classified as water or not water. 

There are two completely sets of functions that do feature extraction in this program. Right now we will only be looking at the ones that are used to classify a video, we will talk about the other ones later. The function that does all the feature extraction is called getFeatures() located in preditctFunctions.py. This function takes in the preprocessed video, the path to a mask if you have one, and the other inputs of preprocessVideo(), in addition it has boxSize, which will be used in getting the temporal features, patchSize and numFramesAvg, which will be used in obtaining spatial features. To calculate the temporal feature, getFeatures() calls fourierTansformFullImage() which is located in the features.py. In this function, you first smooth the image using a couple of filters (kernels are size boxSize), next you perform a Fourier transform on each pixel individually over time, then next an L1 normalization is done so that the absolute amplitude of each pixel doesn’t matter, only the relative amplitudes, so that different images can be compared. Then it returns a 3-D matrix of the Fourier signals in the 3rd dimension.

Then for the spatial feature is created using the SpatialFeatureFullImage() function. First the lbp is taken of each frame of the video. Then for each pixel a sub-box of size patchSize*patchSize*averageFrameNumber is created. Then a histogram of this “cube” is taken and stored in that pixel’s location, then the function returns the matrix.

Next in getFeatures() the two matrices are concatenated so that it creates one matrix on shape (width, height, temporalFeatures + spatialFeatures). Then getFeatures() the concatenated matrix and it also returns a matrix of the mask where the non-water pixel value is 0 and water is 255. 

Going one step up, ModuleC(), which calls getFeatures(), combines the preprocessing step and the feature extraction step into one step.

## Machine Learning Algorithm

The next step is putting the features through the machine learning algorithm. The program is using a random decision forest classifier. This classifier creates several different trees using only some of the features, which alone aren’t always accurate, but a forest of trees tends to be much more accurate because each tree gets a single vote and whichever classification is the majority, is how the image is classified. This is implemented in testFullVid() located in predictFunctions.py. In this function it first obtains the features of the selected video by calling ModuleC() explained above. Then it shapes the feature matrix so that it can be classified. Next it classifies the actual vector which happens in the line `prob = model.predict_proba(newShape)[:,0]`  
From that a mask is created called isWaterFound. After that, I crop the edges because, the features aren’t extracted correctly in the edge cases. Next I regularize the mask using the probabilities that were found from the decision forest. This is done using the Ising Model. This essentially takes a combination of the pixels that surrounds the center pixel and the confidence that the center pixel is correctly classified as water or not-water and uses those two factors to regularize the image. Then the image is cropped again slightly because of the regularization messes with the boundary. Then you return the mask the program created and true mask if you have a mask for the image.

This testFullVid() function is called from inside moduleE(). What moduleE() does, is runs testFullVid() `numVids` amount of times and it will create a final mask where each pixel will be classified as whichever classification was the majority when it ran testFullVid()`numVids` times.


Tl;dr
To run the classifier, all you must do is run moduleE() with the correct parameters.
•	Vidpath – (string) path to the test video or 0 (int)
•	Maskpath - (string) path to the test mask or 0 (int) if you don’t have a mask
•	outputFolder – (string) path to folder where you want the resulting images to output or use “/”to have images output in the current directory (path must exist)
•	numFrames – (int) must be 200 (what current classifier is set to)
•	dFactor – (int) amount of times you want to downsample image (2 is recommended)
•	densityMode – (int) whether density mode is active or direct is activated (0 is recommended – direct mode)
•	boxSize – (int) size of kernel to smooth image
•	patchSize – (int) size of 2-D patch where the Local Binary Pattern will be calculated and histogram taken of
•	numFramesAvg – (int) number of frames of size patchSize which the histogram will be taken of
•	numVids – (int) number of times you want to compute the mask so that you can take the majority amount of pixels



# Training the Classifier

The classifier is trained in trainForest.py. In this file it calls the function LoopsThroughAllVids() which collects the feature vectors and labels (says whether the corresponding feature is water or not water). Then it trains the classifier and saves the model. For later use in the above classifier.

LoopsThroughAllVids() is in containerFunctions.py and takes a certain amount of random sample (numbofSamples) and obtains the feature vectors for each of those points and records whether those points are water or non-water and will then return two long vectors, one that has all the feature vectors concatenated, and one with whether each sample is water or not water.

LoopsThroughAllVids() does this by looping through each folder with sample videos of different situations that you could see water in (streams, ponds, canals etc.). Inside each of these loops it calls moduleB(). Inside moduleB() it calls getFeaturesPoints(). This function choose random points (can choose amount of points per video) in space and time, and extracts the features, both temporal and spatial) of those points and returns a concatenated vector of all the feature vectors you obtained in that video and returns those vector as totalFeatures and whether those features were water or not water in the vector isWater.

Then LoopsThroughAllVids() concatenates all of the totalFeatures from each video together. It also concatenates all the isWater vectors together. This creates one matrix of size (n_samples, feature_length) and a vector of shape (n_samples). Then these two matrices are returned and used to train the classifier in trainForest.py and saves the classifier for later use.
