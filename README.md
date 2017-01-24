## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


In this project, your goal is to write a software pipeline to identify the lane boundaries in a video, but the main output or product we want you to create is a detailed writeup of the project.  Check out the [writeup template](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup.  

Creating a great writeup:
---
A great writeup should include the rubric points as well as your description of how you addressed each point.  You should include a detailed description of the code used in each step (with line-number references and code snippets where necessary), and links to other supporting documents or external references.  You should include images in your writeup to demonstrate how your code works with examples.  

All that said, please be concise!  We're not looking for you to write a book here, just a brief description of how you passed each rubric point, and references to the relevant code :). 

You're not required to use markdown for your writeup.  If you use another method please just submit a pdf of your writeup.

The Project
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

###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is in the P4 Advanced Lane lines notebook. I have started by preparing object points and image points
as indicated in the module videos. I have started by going through all the chessboard images , then I tried finding and drawing the chessboard corners.
Some of the chessboard images my pipeline didn't successfully find and draw the corners and hence wasn't able to warp or transform.

After drawing the chessboard corners and getting the imgpoints and objpoints, I used the OpenCV camera calibration function
with the dist and matrix to calibrate my camera. Then I undistorted my images after the camera calibration
and used the undistorted images to then transform my chessboard images into a flat like image where it appears the camera plan is parallel to the chessboard

After going through the chessboard callibration images, then I took the test images of the road and re passed them throug a similar pipeline
I have not yet though found the correlation between how we callibrated the camera in the step above and the transformation I did with the test images.

After going through the test images, with the help of some students on the forum I was able to get src and dst points
and then I made a birds eye view of the lane lines.

After acquiring the bird view images, then I started working on the following pipeline:

1- Sobel Threshold:
Applied absolute sobel threshold for both `x` and `y` orientation. I have tried using a variety of kernel sizes, and chose a kernel size of 9, which was
slighly higher than 3 and less than 25. I have experimented with a kernal size of 25 and saw a lot of blank images without lane line markings

2- Magnitude Threshold:
After getting `x` and `y` Sobel transformations got the square root magnitude of the Sobel transformation sqaures and then got the gradient
magnitude. I have used kernel size with 9 as well, and applied the threshold input by making all pixels within the threshold as 1 and otherwise zero then
return the binary image.

3- Directional Threshold:
After also getting the `x` and `y` Sobel transformations and getting the arc tan value, I applied a kernel size of 9 and
a threshold from 20 to 180 degrees. I feel I have failed in the directional threshold to find the right hyper parameters
I am still seeing a blank image for the directional threshold transformation and not sure where the missing piece is.

After visualizing those thresholds, I have took the lecture's recommendation and wanted to visualize how the combined threshold would look like.

It seems throughout my visualizations I am suffering from a variety of noise levels and don't have the right hyper parameters yet.. It also seems
having the same hardcoded hyper parameters for all images is not a good strategy since they have different levels of lighting

Not sure yet what would be the best to computer the different hyper parameters.

After applying the magnitude thresholds, I took a look at using a different color space: HLS

I have converted my image into an HLS image, and then separate each of the H, L and S channels into their own arrays.
After separating the three channels applied the threshold on the image from the S channel , since given from the excercies throughout the module
we have been getting better results with the S channel.

After all those transformations it then made sense to apply the pipeline provided in the lesson which applied more than one
system of color channels on the image where I got a better result that made my lane lines clearer.


In conclusion this is still a work in progress, and I am still working on calculating the curvature and then applying it on the video

The images for camera calibration are stored in the folder called `camera_cal`.  The images in `test_images` are for testing your pipeline on single frames.  To help the reviewer examine your work, please save examples of the output from each stage of your pipeline in the folder called `ouput_images`, and include a description in your writeup for the project of what each image shows.    The video called `project_video.mp4` is the video your pipeline should work well on.  

The `challenge_video.mp4` video is an extra (and optional) challenge for you if you want to test your pipeline under somewhat trickier conditions.  The `harder_challenge.mp4` video is another optional challenge and is brutal!

If you're feeling ambitious (again, totally optional though), don't stop there!  We encourage you to go out and take video of your own, calibrate your camera and show us how you would implement this project from scratch!
