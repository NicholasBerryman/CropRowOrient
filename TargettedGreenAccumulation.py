# Test for Bengochea-Guevara et al. Crop Detection

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
imageLarge = cv2.imread("barley/"+imgName)
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


# Sobel filter
xFilter = cv2.Sobel(filtered, cv2.CV_64F, 1, 0, ksize=3)
xAbs = cv2.convertScaleAbs(xFilter)
yFilter = cv2.Sobel(filtered, cv2.CV_64F, 0, 1, ksize=3)
yAbs = cv2.convertScaleAbs(yFilter)
sobel = cv2.addWeighted(xAbs, 0.5, yAbs, 0.5, 0)


# Row counting for approximate potential centre line
rows = np.vsplit(filtered, 480)
count = 0
potentialRow = -1
i = 0
for row in rows:
	newCount = cv2.countNonZero(row)
	if (newCount > count):
		count = newCount
		potentialRow = i 
	i += 1


#Split image into segments
nCols = 1
xOffset = 640/nCols
imageParts = list()
for i in range(0,nCols):
        imageParts.append(filtered[0:480, i*(int)(640/nCols):i*(int)(640/nCols)+(int)(640/nCols)])
i = 0
for c in imageParts:
        i+=1
        cv2.imshow("cropped"+str(i),c)

# Defining our window
windowWidth = 200
windowHeight = 75
windowOffsetY = 0
windowOffsetX = 0
windowMinX = int(640/2 - windowWidth - windowOffsetX)
windowMaxX = int(640/2 + windowWidth - windowOffsetX)
windowMinY = int(480/2 - windowHeight - windowOffsetY)
windowMaxY = int(480/2 + windowHeight - windowOffsetY)



# Contour detection
i = 0
allContours = list()
for part in imageParts:
        contours, heirarchy = cv2.findContours(part, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
                cx, cy = getContCentre(contour)
                if (cx > windowMinX and cx < windowMaxX and cy > windowMinY and cy < windowMaxY): #TODO check distance (this is not d2 - not quite sure what it should be)
                        contour += [int(xOffset * i), 0]
                        allContours.append(contour)
        i+=1

# Get centres
rowXs = list()
rowYs = list()
for contour in allContours:
        cx, cy = getContCentre(contour)
        rowXs.append(cx)
        rowYs.append(cy)

# Do regression on plant coordinates
rowLinear = LinearRegression().fit(np.array(rowXs).reshape((-1,1)), np.array(rowYs))
r2 = rowLinear.score(np.array(rowXs).reshape((-1,1)), np.array(rowYs))
intercept = rowLinear.intercept_
final = rowLinear.intercept_ + 640 * rowLinear.coef_
print("R^2: "+str(r2))

# Draw Images
cv2.line(image, (0, int(intercept)), (640, int(final[0])), (0,255,255))
cv2.rectangle(image, (windowMinX, windowMinY), (windowMaxX, windowMaxY), (255,0,0), 1)
i = 0
for x in rowXs:
        cv2.circle(image, (x, rowYs[i]), 3, (0,255,0))
        i += 1
        
cv2.imshow("linear combination", linearCombine)
cv2.imshow("threshold", threshImage)
cv2.imshow("filtered", filtered)
cv2.imshow("sobel", sobel)
cv2.imshow("original", image)


cv2.waitKey(0)

