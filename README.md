# Advanced Lane Finding Project
---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"


[test1]: ./examples/test_1.png "Test image"
[test1undistort]: ./examples/test_1_undistort.png "Test image undistorted"
[test1absbinary]: ./examples/test_1_abs_binary.png "Test image abs grad in x"
[test1dirbinary]: ./examples/test_1_dir_binary.png "Test image direction grad"
[test1magbinary]: ./examples/test_1_mag_binary.png "Test image magnitude grad"
[test1sbinary]: ./examples/test_1_s_binary.png "Test image s channel"
[test1combinedbinary]: ./examples/test_1_combined_binary.png "Test image combined binary"

[test1warped]: ./examples/test_1_warped.png "Perspective transformed image"
[test1processed]: ./examples/test_1_processed.png "Image after pipeline"
[test1histogram]: ./examples/test_1_histogram.png "Histogram for line lane finding"
[test1lanesdetected]: ./examples/test_1_lanes_detected.png "Left and right lanes found"
[test1endresults]: ./examples/test_1_endresults.png "Lane detected"
[test1visualizedfit]: ./examples/test_1_visualizedfit.png "visualized fit"

---

## Camera Calibration

#### Briefly state how you computed the camera matrix and distortion coefficients. 

The code for this step is contained in the first code cell of the IPython notebook located in "lane_lines.ipynb"

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  The objpoints and imgpoints can be obtained by using the `compute_cal_points` function in the very first cell.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function.  The result can be seen below in the discussion of the pipeline used for processing the images.

## Pipeline

The entire pipeline for processing images consists of the following:
* correcting a distorted image
* performing binary thresholding
* perspective transform

I decided to use a combination of gradients and color transforms to achieve a threshold binary image.  I used the direction gradient, the magnitude gradient, the absolute x gradient, and the s channel in the hls color space.

#### Original Image
![alt text][test1] 

#### After a distortion correction
![alt text][test1undistort]

### Here we will display the following color transforms and gradient binary images.

#### Absolute gradient in x
![alt text][test1absbinary]

#### Direction gradient
![alt text][test1dirbinary]

#### Magnitude gradient
![alt text][test1magbinary]

#### S channel
![alt text][test1sbinary]

#### And finally, here is the combined binary image
![alt text][test1combinedbinary]

## Perspective transform 

The code for my perspective transform includes a function called `corners_unwarp(image, mtx, dist)`.  Here I use `cv2.getPerspectiveTransform` and `cv2.warpPerspective` to get a bird's eye view of the road.  The source and destination points were hard coded and taken from the Udacity Slack Channel.

```python
    # Points were found on the Udacity Slack Channel by Chris Grill
    src = np.float32([[220, 719], [1220, 719], [750, 480], [550, 480]])
    dest = np.float32([[240, 719], [1040, 719], [1040, 300], [240, 300]])
```
#### Original
![alt text][test1]

#### Here is the image after the perspective transform
![alt text][test1warped]

## Identification of lane-lines

#### Here is the image after the entire pipeline
![alt text][test1processed]

At this point, we want to determine the pixels that make up the left and right line lanes.  After that we also want to fit a curve to these lines.

One of these methods makes use of a histogram.  We use a histogram to determine the peaks on the left and right side of the midpoint.  This is also done with the bottom half of the image.  We then use a sliding window to get all the points in the lane-lines.  This can all be seen under the fit_lines(img, histogram) function under "Detecting the left and right lanes" in the lane_lines.ipynb.  Once we have enough points, we can fit a 2nd degree polynomial to the lines.

Once we know where the lines are, we do not need to start over and perform a blind search with a histogram.  We can instead search around in a margin around the previous line position.  This can be seen in `fit_lines-from_prev` in the notebook.

A Line class is also used to keep track of the past n fits.  In this case, I chose to use 10 for n.  This smooths out the area of interest because we are using an average of the previous fits.  If we cannot find enough pixels, then we should have a good enough history to keep going on the lane, but at this point we would restart with the histogram method.  

#### Histogram of peaks
![alt text][test1histogram]

#### Visual of the lines and identified pixels
![alt text][test1lanesdetected]

#### Visual of the search from a previously identified line
![alt text][test1visualizedfit]


## Radius of curvature of the lane and the position of the vehicle

The code for calculating curvatures is in a function called calculate_curvatures under the section "Calculating left and right curvatures" in the lane_lines.ipynb
The code was essentially provided by Udacity, but I will briefly discuss the calculations.  The function takes in 
the equation of the lane lines and the lane pixel indices.

We have polynomials that fit a line, which are in the form f(y) = Ay^2 + By + C
The equation for radius of curvature requires taking the derivative and second derivative of f(y).

f'(y) = 2Ay + B 

f''(y) = 2A

R_curve = (1 + (2Ay + B)^2)^(3/2) / abs(2A)


Under the same section, you will see `distance_from_lane_center(img, left_fit, right_fit)` which will return the position of the vehicle with respect to the center.

For this, we use the center of the image as our camera position, and we want to find out the distance from our camera position to the center of the lane.  The center of the lane is the midpoint of the two lane lines at the very bottom of the curve.  A positive number indicates that we are on the right side of the lane center nad a negative number indicates that we are on the left side of the center.

#### Here are the results of the lane being correctly detected

![alt text][test1endresults]

---

## Pipeline (video)

Here's a [link to my video result](./project_video_output.mp4)

---

## Discussion

####  Discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

After going through the challenge video, it seems that the barrier in the middle can be picked up as another line.  It also has problems with cracks in the road as it seems to prefer the crack over the dashed lines.  To make the pipeline more robust, we could add a deep learning approach for detecting lane lines.  If the convolutional network we implement is trained correctly, then this would ensure that only lane lines get picked up. 