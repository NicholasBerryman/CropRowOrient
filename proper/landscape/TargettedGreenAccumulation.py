import numpy as np
import time
import cv2
from sklearn.linear_model import LinearRegression
import random
import colorsys


def getContCentre(contour):
        M = cv2.moments(contour)
        cx = 0
        cy = 0
        if (M['m00'] != 0):
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
        return cx, cy

# grab the raw NumPy array representing the image, then initialize the timestamp
# and occupied/unoccupied text
imgName = input("File Name: ")
imageLarge = cv2.imread(imgName)
image = cv2.resize(imageLarge, (640,480))

# Convert to greyscale with custom linear combination (From source)
greyCoeffs = [-0.311, 1.262, -0.884] #BGR
coeffNP = np.array(greyCoeffs).reshape((1,3))
linearCombine = cv2.transform(image,coeffNP) #Greyscale image


# Threshold from average
###threshImage = cv2.adaptiveThreshold(linearCombine, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 0)
ret, threshImage = cv2.threshold(linearCombine, 0, 255, cv2.THRESH_OTSU)


# Filter for enhancement
kernel = np.ones((3,3), np.uint8)
filtered = cv2.morphologyEx(threshImage, cv2.MORPH_OPEN, kernel) #Erodes then dilates

# Defining our window
windowWidth = 100
windowOffset = -50

# Row counting for approximate potential centre line
cols = np.hsplit(filtered, 640)
count = 0
potentialRow = -1
i = 0
for row in cols:
	if (i > 640/2 - windowWidth + windowOffset and i < 640/2 + windowWidth + windowOffset):
		newCount = cv2.countNonZero(row)
		if (newCount > count):
			count = newCount
			potentialRow = i 
	i += 1



# Draw Images
cv2.line(image, (int(potentialRow), 0), (int(potentialRow),480), (0,255,255), 3)
i = 0
        
cv2.imshow("linear combination", linearCombine)
cv2.imshow("threshold", threshImage)
cv2.imshow("filtered", filtered)
cv2.imshow("original", image)


cv2.waitKey(0)

