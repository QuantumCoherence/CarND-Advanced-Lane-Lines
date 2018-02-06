

#**Advanced Lane Finding Project**

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

[image0]: ./Find_Corners_calibration_images.png "Calibration Images"
[image1]: ./calibration_distortion_correction.png "Calibration Undistortion"
[image2]: ./Lane_Lines_thresholding_study_sample.png "Thresholding Samples"
[image3]: ./straight_lines.png "Straight Lane Lines"
[image4]: ./thresholded_binary_straight_lines.png "Thresholded Straight Lane Lines"
[image5]: ./warped_birds_eye_view_of_lane_lines.png "Warped Bird's Eye view"
[image6]: ./lane_lines_warping_study_sample.png "Bird's eye warping study samples"
[image7]: ./sliding_windows_sample1.png "Line Detection Sample 1"
[image8]: ./sliding_windows_sample2.png "Line Detection Sample 2"
[image10]: ./Lane_unwarping_and_overlaying.png "Lane_unwarping_and_overlaying"
[image9]: ./Curvature_and_lane_position_Code.png "Curvature_and_lane_position_Code"
[image11]: ./pipeline.png "Pipeline code"
[image12]: ./project_video_13.png "Project Video Frame"
[image13]: ./project_video_33.png "Project Video Frame"
[video1]: ./projectVideoOut.mp4 "Output Video"

---


## 1. Camera Calibration/ Distortion Correction
**Points 1 and 2 of the jupyter notebook - Cells 1 to 2**

#### 1. Find Corners of chessboard calibration images.

See the code in Cell 1 of the attached jupyter noteboook

1. Define reference points objp 
2. For all calibration images
	1. Find corners
	2. Draw line joining all found corners
	3. assign corner coordinates to reference objpoints

![alt text][image0]

#### 2. Estimate Calibration and undistort matrices.

See the code in Cell 2 of the attached jupyter noteboook

1. Using corners coordinate and reference points collected under 1. estimate calibration matrix
`	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)`  
2. For all calibration, perform undistort to test result
`	dst = cv2.undistort(img, mtx, dist, None, mtx)`
See sample output below 
 
![alt text][image1]

## 2. Lane Lines Image Processing

**Point 3 of the jupyter notebook - Cells 3 to 6**

A series of frames were extracted from all the project videos to be used to study what combination of threshold image preprocessing algorithms would perform best with extracting lane line features. 

The implementation code of these functions is to be found in the notebook cell number 3.


**def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0,255))**

     returns two images:
     The scaled sobel derviative on either x or y direction 
	 and 
     The binary ouput of the sobel derviative on either x or y direction, 
     thresholded according to the threshold parameter tresh

**def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255))**

    Returns the sobel derivative magnitude thresholded binary image accroding 
	to the  thresold and sobel kernel depth parameters  
    

**def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2))**

	Retruns the sobel gradient magnitude thresholded according to the 
	given threshold direction parameters in radiants 
    and to the sobel kernel depth
 
**def filterHorizontalLines(image, kernel_size = 5,iterations=1)**

	Returns the input image filterd of any horizontal line longer than the kernel_size paramter   

**def hls_select(img, thresh=(0, 255), channel = 's')**

    returns the selected HLS channel (either 'h', 'l' or 's') and the binary 
	thresholded version of the slected channel,according to the given threshold parameter 

**def hsv_select(img, thresh=(0, 255), channel='s')**

	returns the selected HSV channel (either 's', 'h' or 'v')


**def colorfilter(image)**

    returns three images with only respectively the white, yellow and black color components  


After various image processing studies, as illustrated in the following image with a sample of one of many studies
  
![alt text][image2]

the following preprocessing was selected that includes undistortion and bilateralFilter smoothing. 

The full study is in cell 5 and the final image_thresh function is in cell 6.

    def image_thresh(image, mtx, dist):
    # Use the OpenCV undistort() function to remove distortion
    undist_image = cv2.undistort(image, mtx, dist, None, mtx)
    
    sobel, sobelbin = abs_sobel_thresh(undist_image, orient='x', sobel_kernel=5,thresh=(18, 126))
    
    # S-Channel
    s_channel, schxbin = hls_select(undist_image, thresh=(90, 255), channel = 's')
    schxbin = filterHorizontalLines(schxbin, kernel_size = 35, iterations = 1)    

    # Colorextraction
    white, yellow, black = colorfilter(undist_image)

    # Combine s-channel and sobel-x with smoothing and H-Filter
    smoothcomb = cv2.bilateralFilter(schxbin | sobelbin,9,75,75) 
    cleancomb = filterHorizontalLines(smoothcomb, kernel_size = 35, iterations = 1)    
    
    # Add smoothed Yellow color 
    smoothyellow = cv2.bilateralFilter(yellow,9,75,75) 
    processsed_image = np.zeros(undist_image.shape[:2])
    processsed_image[((cleancomb == 1) | (smoothyellow == 1))] = 1
    
    return processsed_image, undist_image


## 3. Bird-Eye's View Transform of Lane Images
**Point 4 of the jupyter notebook - Cells 7 to 9**

The code for the perspective transform includes a function called `unwarp(pimg, src, dst)`, which appears in cell number 7 and takes as input an image (`img`), as well as source (`src`) and destination (`dst`) points.

A note about engineering and the trial and error process demonstrated in the lesson on how to select the src and dst points. Please do better... this is not engineering. The method is totally unreliable and cannot be applied on the other videos without having to modify the src and destination points. 
  
The following images show respectively the original image, the thresholded binary version and the bird's eye view of the lane lines. 

![alt text][image3]

Please note the red and black boxes represent the src and dst for the perspective transform points. Their representation on this image is only approximated.
 
![alt text][image4]
Threshodled image
 
![alt text][image5]
Bird's-eye view

Cell number 9 include a full study of the Bird's eye perspective transform. Here below a sample of the study

![alt text][image6]

## 4.Algorithmically Extract Lane Lines
**Point 5 of the jupyter notebook - Cells 11 to 18**

The algorithm for lane line extraction uses the sliding window method as described in the lesson. 
Once the pixel points of the left and right lane are detected using a histogram analysis of the image, a polynomial fit is applied to the detected pixel  points. The outcome is then used to calculate the actual fitted points of the lane lines, to be passed for further plotting back onto the original video.

![alt text][image7]
![alt text][image8]

The above figures show the outcome of the sliding windows method for two subsequent frame sfrom  the Bird's eye study under the previous section.

The code for this lane line detection is in cells 14 to 18. 
Cells 11 to 13 show the code for the convolution method, which is more efficient but more prone to errors.

## 5. Estimate Lane Curvature and Vehicle Position
**Point 6 of the jupyter notebook - Cell 19**

The following code snippet shows the curvature and position estimation algorithms 

![alt text][image9]


## 6. Warp estimates back to original view and Visually Overlay detected Lane onto video
**Points 7 and 8 of the jupyter notebook - Cell 20**

Following image shows the code used to warp the fitted lane lines back onto to the original video frame 
![alt text][image10]
 
## 7. Pipeline (video)
**Point 9 of the jupyter notebook, Cells 21 to 28**



    def process_image(image)

This the function provided under point 9 of the jupyter notebook, cell number 26, that implements the full algorithm and is passed to the videoclip function.

![alt text][image11]

Cell 23 contains a "Frame overalying study", of which you can see two sample images here below

![alt text][image12]
![alt text][image13]

## 8. The project video 
Here's a [link to video result](./projectVideoOut.mp4)

---

### Discussion


It is quite obvious that thresholding and processing images taken in the visual light range of frequencies will never be able to cover 100% of the light conditions a vehicle's camera system will encounter on the road, while trying to detect lane lines. 
One simple example is the camera being directly into direct sunlight, which would then saturate the sensor, regardless of compensation techniques. Even sophisticate sensors cannot always filter all possible light conditions. 
  
From these videos we can easily see, that in some occasions the road condition simply do not provide  lane lines information. It begs the question ..  are lane absent because the road was washed away by a landslide or simply because the light condition don't allow an easy lane line detection?

If we ignore bad light condition and extrapolate blindly from frame to frame , how do we guarantee that by ignoring missing lane lines, we are also not ignoring a fully missing lane, simply because there is no road pavement?

So, the question becomes then .. what is a lane finding algorithm supposed to detect? Just lane lines or if there is a road in front of the vehicle? Can camera sensor only be reliable enough to detect all what is required to be detected.

A quick research on academic work in this field, shows a large body of work and quite sophisticated published algorithm that show, this advanced lane lines detection method barely scratched the complexity  of lane lines and in general lane detection problems.  


