
**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.png
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.png
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video_processed.mp4
[video2]: ./project_video_car_lane_detected.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

### All codes are in vehicle_detection.ipynb.

---

## 1. Writeup / README

You're reading it!

## 2. Feature extraction
Three types of features were extracted and used in this project:
1. Histogram of Oriented Gradient (HOG)feature: "The technique counts occurrences of gradient orientation in localized portions of an image", see [wiki](https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients)
2. Binned color feature: Perform spatial binning on an image and  retain enough information to reduce the features extracted from raw pixels
3. Color histogram feature: Use raw pixel intensities as features

### 2.0 Datasets

I used both GTI vehicle image databases and the KITTI vision benchmark suite. I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]


### 2.1 Histogram of Oriented Gradients (HOG)
 > Code cell #6.


I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `RGB` color space, CH-0, and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]




#### Explain how you settled on your final choice of HOG parameters.


I tried various combinations of parameters and chose the HOG parameters of orientations=9, pixels_per_cell=(8, 8) and cells_per_block=(2, 2).

### 2.2 Binned Color Feature
 > Code cell #7.

I also tried getting the binned color feature by scaling down the resolution to 32 by 32. The code is in code cell #11.


### 2.3 Color Histogram Feature

 > Code cell #8.

Histograms of pixel intensity (color histograms) are also helpful for distinguishing car from non-car. The code is in 12th cell. The default ranges of bins is 0 to 255. However, I used a range of (0, 1) when I read in an image in PNG format using `mping`(e.g. in cell 13).


**In sum, I extracted 6108 features to train the model.**

## 3. Trained a classifier using your selected HOG features and color features
### 3.1 Prepare training and test sets.
 > Code cell #12.

I kept 20% of the images for testing. I normalized the feature vectors using `sklearn.preprocessing.StandardScaler()` in code cell #16. I also shuffled the data points when splitting them into train set and test set. Eventually, there were 14208 samples for training and 3552 for testing.

### 3.2 Train a classifier

 > Code cell #13.

I trained a simple linear SVM using `sklearn.svm.LinearSVC` and managed to get a test accuracy of 98.69%. I did try `sklearn.svm.svc`, which gave a 99.1% test accuracy but was way slower than the `linearSVC`.

I used the `LinearSVC` classifier to detect cars.

## 4. Used Sliding Window Search to find cars

### 4.1 Interests of area, scale and overlap windows

 > Code cell #14.

I focused the lower half of the frame where the cars would show. I used three scales(1, 1.5, and 2) to roughly cover small(far), medium(middle), and large(close) vehicles in the image.  Instead of setting overlap directly, I moved the windows by 2 pixels, vertically or horizontally, in each step, which gave a overlap rate of 75%.


### 4.2 Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

 > Code cell #15.

Ultimately I searched on three scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

## 5. Video Implementation

### 5.1. Final video output.

Here's a [link to my video result for car detection.](./project_video_processed.mp4)
I also added lanes to the video, here is a [link to the video for both car and lane detections](./project_video_car_lane_detected.mp4).


### 5.2 Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

 > Code cell #16.

I recorded the positions of positive detections in each frame of the video. I  integrated a heat map over several frames of video, such that areas of multiple detections get "hot", while transient false positives stay "cool". I then thresholded that map to identify vehicle positions.  I used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  


Here's an example result showing the results of the pipleline working on a single frame (In the project, I actually integrated heatmap from multiple frames. I just didn't find an easy way to export sequential images from video to show the work).

![alt text][image5]


---

## Discussion

A summary of the project:
- Datasets: GTI vehicle image databases and the KITTI vision benchmark suite, 14208 samples for training and 3552 for testing.
- Features: Histogram of Oriented Gradient (HOG), Binned color feature, and Color histogram feature, 6108 features were extracted.
- Model: linearSVC, test accuracy: 98
- Car detection strategy: sliding windows + thresholded heatmap on multiple frames

Potential issues:
1. Speed. Current algorithm used to detect cars in the frame is quite time consuming. It took more than 20 minutes to process a 50-seconds video. Since a car cannot go too far in a few milliseconds, I can speed the search up by reducing the interests of areas for searching.

2. Failed to detect car occasionally. In the example video, things got hard when the white car drove on gray pavement. Two possible reasons: 1) it's more difficult to distinguish a car from non-car when its color is close to the color of the non-car objects 2) there were not many white cars in the training set. To address the first issue, I may leverage the shape features (like tuning HOG parameters) and use less color features. For the second issue, I can simply collect more images of white cars for training.
