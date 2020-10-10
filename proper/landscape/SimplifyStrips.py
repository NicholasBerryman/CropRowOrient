
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

#Split image into segments
nRows = 5
yOffset = 480/nRows
imageParts = list()
for i in range(0,nRows):
	imageParts.append(filtered[i*(int)(480/nRows):i*(int)(480/nRows)+(int)(480/nRows), 0:640])
i = 0
for c in imageParts:
	i+=1
	#cv2.imshow("cropped"+str(i),c)


# Defining our window
windowWidth = 100
windowOffset = -50
plantWindowWidth = 75

# Column counting for approximate potential centre line
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

#Finding max proportion in strip columns
maxProp = 0
for c in imageParts:
        iCol = 0
        for col in c.T:
                proportion = cv2.countNonZero(col)/len(col)
                if proportion >= maxProp and iCol > 640/2 - windowWidth + windowOffset and iCol < 640/2 + windowWidth + windowOffset:
                        maxProp = proportion
                iCol += 1
                        
#Filing out satisfactory strip columns, culling unsatisfactory
satisThreshold = 0.6
i = 0
rowXs = list()
rowYs = list()
for c in imageParts:
        iCol = 0
        for col in c.T:
                proportion = cv2.countNonZero(col)/len(col)
                if proportion >= satisThreshold*maxProp and iCol > 640/2 - windowWidth + windowOffset and iCol < 640/2 + windowWidth + windowOffset:
                        c[:,iCol] = 255
                        rowXs.append(iCol)
                        rowYs.append((int)((i+0.5)*(int)(480/nRows)))
                else:
                        c[:,iCol] = 0
                iCol += 1
                        
        i += 1
                        
#Restitch image
restitched = imageParts[0]
for c in imageParts[1:]:
        restitched = np.concatenate((restitched, c), axis = 0)



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
cv2.imshow("restitched", restitched)
cv2.imshow("original", image)


cv2.waitKey(0)

