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
image = cv2.imread(imgName)

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


# Column counting for green pixel proportions (candidate row)
windowWidth = 100 #Doubled in search
minProportion = 0.2
maxCol = 640-200

cols = np.hsplit(filtered, 640)
colProps = []
i = 0
candidateRow = -1
for col in cols:
        colProps.append(cv2.countNonZero(col)/640.0)
        if (colProps[i] > minProportion and i < maxCol):
                candidateRow = i
        i += 1
	#print(colProps[i-1])

# Find rightmost edge of crop row
rowXs = list()
rowYs = list()
rightPixels = filtered.copy()
y = 0
for row in range(0,480):
        foundEdge = False
        x = candidateRow + windowWidth
        while x >= candidateRow - windowWidth:
                px = rightPixels[y,x]
                if px > 0 and not foundEdge:
                        foundEdge = True
                        rowXs.append(x)
                        rowYs.append(y)
                        pass
                else:
                        rightPixels[y, x] = 0
                x-=1
        y += 1

cv2.imshow("binary", filtered)


# Do regression on plant coordinates
rowLinear = LinearRegression().fit(np.array(rowYs).reshape((-1,1)), np.array(rowXs))
r2 = rowLinear.score(np.array(rowYs).reshape((-1,1)), np.array(rowXs))
intercept = rowLinear.intercept_
final = rowLinear.intercept_ + 640 * rowLinear.coef_
print("R^2: "+str(r2))

# Draw Images
cv2.line(image, (int(intercept), 0), (int(final[0]), 480), (0,255,255), 3)
i = 0
for x in rowXs:
        cv2.circle(image, (x, rowYs[i]), 3, (0,255,0))
        i += 1
        
cv2.imshow("linear combination", linearCombine)
cv2.imshow("threshold", threshImage)
cv2.imshow("filtered", filtered)
cv2.imshow("cropEdge", rightPixels)
cv2.imshow("original", image)


cv2.waitKey(0)

