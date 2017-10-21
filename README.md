# Advanced Lane Lines (Finding the lane lines on the road)
In this project, I have used the OpenCV to detect the lane lines on the road from a video. A more detailed explanation will be provided below. The final video can be seen here: https://youtu.be/jXlH0ru9PNE

# Required options

  ```sh
$ Keras==1.2.1
$ numpy==1.13.1
$ matplotlib==2.0.2
$ ipython==6.1.0
$ pandas==0.20.3
$ tensorflow==0.12.1
```




## Installation

In order to detect the lane lines from your own video, you can simply upload to the local file and change the **clip1** variable file path to yours in the last code cell. 
*Note* : The lane line detection might not work perfectly on your own video, as the warping coordinates have been hardcoded.

# How it works in short
**Camera Calibration**:
The camera calibration is happening in the code cell 2. The first thing I did, was to prepare the object points which are coordinates in 3 dimensions (x,y,z). Due to the nature of the images, the z value is always 0. These fixed coordinates are stored in the list called objp. The chessboard image has 9x6 points.
So, taking distorted samples of the chessboard images, whenever the chessboard detection happens, objp list is automatically appended to the objpoints list. Also, during this process, the imgpoints list, which are the pixel coordinates of the distorted images are also detected. Thus, having two values, object points (essentially the goal points) and the image points (the pixel locations of the chessboard), using the cv2.calibrateCamera() function, the camera has been calibrated. After investigating all the samples and calibrating the camera, the next step was to undistort the image, using the cv2.undistort() function and test the result.

**Creating a threshold binary image**:
I used a combination of colors (R and G to identify the yellow lines) and gradient thresholds (S and L from the HLS, to identify the bright white lines and to eliminate any shadows) to generate a binary image (in code cells 4 and 5).

**Warping the image**:
After applying a threshold the next step was to perform a perspective transform, by firstly finding the M value from cv2.getPerspectiveTransform(src,dest) and also M_inv in order to unwarp at the final stage, by simply changing the src and dest with each other. For the source (src) points, I have chosen the values like:
[200,720], [599,446], [680,446], [1100,720]
Then, the destination points are:
[200+offset,720], [200+offset,0], [1100-offset,0], [1100-offset,720], offset being 200.

**Finding the lane lines**:
In order to find the lane lines, I have squashed all the binary pixels using the np.sum and found the spikes. Those spikes (x values of the spikes) were the starting points of the lane lines (left and right).
After that, I have divided the image into 9 portions (9 windows) from the Y perspective. Within those 9 windows I have started to search for the pixels by starting from the starting point and only searching within +/- 100 pixels across X axis. If the pixels were found (minimum 50 pixels to satisfy), the next iteration (next window) will start from taking the average of the location from the identified pixels (x axis values). If less than 50 pixels were fond, the next iteration will continue at previous x axis value.

**Calculating the radius curvature**:
To measure the curvature for the each line, I have used the coefficients values from my polynomial fits and then fitted again but this time, against the empty y values (0-719)
After finding the y positions, I have fitted the polynomial and then applied the formula, for finding the radius of curvature.

**Not perfect**:
The biggest issue I have experienced is with shadows. I have tried numerous experiments by picking the right thresholds and removed most of the shadows by using the L value from the HLS spectrum. Still, it is not perfect and if the shadow will exist for a longer distance, my detection will probably fail.
The improvement here could be to use the previously saved polynomials in order to approximate the lanes and use better thresholds or different spectrum to remove the shadow completely.
I have edited the spectrum and used R & G or L(lum) or H(hls) and tuned it to the best output. Sometimes, when the right lane is not able to fully detect the lane, it wobbles a bit and I have tried to average, use the previous polynomial, adding the midpoint values to the left values in order to construct a virtual polynomial and even measured the slope of both lines to detect the unusual lanes. However, I did saved the polynomial coefficients for the right lane if the first coefficient is smaller than 1e-4, which I have observed produces the best results. Therefore, if for some reason, the right lane will get out of control and its first polynomial coefficient is larger than 1e-5, then the code just takes the previous polynomial coefficients.
