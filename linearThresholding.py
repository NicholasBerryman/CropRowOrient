# Test for Bengochea-Guevara et al. Crop Detection

import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (640,480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))
# allow the camera to warmup
time.sleep(0.1)

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	# grab the raw NumPy array representing the image, then initialize the timestamp
	# and occupied/unoccupied text
	image = frame.array


	# Convert to greyscale with custom linear combination (From source)
	greyCoeffs = [-0.311, 1.262, -0.884] #BGR
	coeffNP = np.array(greyCoeffs).reshape((1,3))
	linearCombine = cv2.transform(image,coeffNP) #Greyscale image


	# Threshold from average
	threshImage = cv2.adaptiveThreshold(linearCombine, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 0)


	# Filter for enhancement
	kernel = np.ones((3,3), np.uint8)
	filtered = cv2.morphologyEx(threshImage, cv2.MORPH_OPEN, kernel) #Erodes then dilates


	# Sobel filter
	xFilter = cv2.Sobel(filtered, cv2.CV_64F, 1, 0, ksize=3)
	xAbs = cv2.convertScaleAbs(xFilter)
	yFilter = cv2.Sobel(filtered, cv2.CV_64F, 0, 1, ksize=3)
	yAbs = cv2.convertScaleAbs(yFilter)
	sobel = cv2.addWeighted(xAbs, 0.5, yAbs, 0.5, 0)














	edgeImg = cv2.Canny(image, 80, 220)
	contours, heirarchy = cv2.findContours(edgeImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	#cv2.drawContours(image, contours, -1, (0,255,0), 3)

	# merge each channel again
	#merged = cv2.merge([B, G, R])
	cv2.imshow("linear combination", linearCombine)
	cv2.imshow("threshold", threshImage)
	cv2.imshow("filtered", filtered)
	cv2.imshow("sobel", sobel)
	cv2.imshow("original", image)


	# clear the stream in preparation for the next frame
	rawCapture.truncate(0)


	key = cv2.waitKey(1) & 0xFF

