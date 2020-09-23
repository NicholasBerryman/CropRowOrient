# Test for Bengochea-Guevara et al. Crop Detection

import numpy as np
import time
import cv2
from sklearn.linear_model import LinearRegression
import random
import colorsys

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
#threshImage = cv2.adaptiveThreshold(linearCombine, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 0)
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
nCols = 3
imageParts = list()
for i in range(0,nCols):
        imageParts.append(filtered[0:480, i*(int)(640/nCols):i*(int)(640/nCols)+(int)(640/nCols)])
i = 0
for c in imageParts:
        i+=1
        cv2.imshow("cropped"+str(i),c)

# Contour detection to perform distance filtering (similar, but not same as in paper)
eachPlantContours = list()
eachPlantIndex = list()
contPlantDict = {}
croppedIndex = 0
rowXs = []
rowYs = []
for part in imageParts:
        contours, heirarchy = cv2.findContours(part, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Iterate over each contour and remove any that cannot be reached by hops of < size 'd2' from contours that intersect our potential row
        inRangeToPotential = list()
        inIndices = list()
        i = 0
        for cont in contours:
                moment = cv2.moments(cont)
                #xCont =int(moment["m10"] / moment["m00"])
                if (moment["m00"] != 0): yCont = int(moment["m01"] / moment["m00"])
                if (abs(potentialRow - yCont) < 20): #TODO check distance (this is not d2 - not quite sure what it should be)
                        inRangeToPotential.append(cont)
                        inIndices.append(i)
                i += 1

        for cont in inRangeToPotential:
                moment = cv2.moments(cont)
                if (moment["m00"] != 0): xCont =int(moment["m10"] / moment["m00"])
                if (moment["m00"] != 0): yCont = int(moment["m01"] / moment["m00"])
                i = 0
                for contCompare in contours:
                        if (i not in inIndices):
                                momentComp = cv2.moments(contCompare)
                                if (momentComp["m00"] != 0): xContComp = int(momentComp["m10"] / momentComp["m00"])
                                if (momentComp["m00"] != 0): yContComp = int(momentComp["m01"] / momentComp["m00"])
                                distance = ((abs(xCont-xContComp))**2.0+(abs(yCont-yContComp))**2.0)**0.5
                                if (distance < 30): #TODO check (this is d2)
                                        inRangeToPotential.append(contCompare)
                                        inIndices.append(i)
                        i += 1
        # Get initial regression line (for comparison, not actually used)
        for cont in inRangeToPotential:
                moment = cv2.moments(cont)
                rowXs.append(int(moment["m10"] / moment["m00"]) + (croppedIndex*(int)(640/nCols)))
                rowYs.append(int(moment["m01"] / moment["m00"]))
        rowLinear = LinearRegression().fit(np.array(rowXs).reshape((-1,1)), np.array(rowYs))
        r2 = rowLinear.score(np.array(rowXs).reshape((-1,1)), np.array(rowYs))
        initialIntercept = rowLinear.intercept_
        initialReg = rowLinear.intercept_ + 640 * rowLinear.coef_
        print("R^2: "+str(r2))

        # Iterate over each contour and label any contours that can be reaced by hops of < size 'd1' as belonging to the same plant
        j = 0
        for cont in inRangeToPotential:
                moment = cv2.moments(cont)
                xCont =int(moment["m10"] / moment["m00"])
                yCont = int(moment["m01"] / moment["m00"])
                i = 0
                if (j not in eachPlantIndex):
                        eachPlantContours.append(list())
                        contPlantDict[str(j)] = len(eachPlantContours)-1
                for contCompare in inRangeToPotential:
                        if (i not in eachPlantIndex):
                                momentComp = cv2.moments(contCompare)
                                xContComp = int(momentComp["m10"] / momentComp["m00"])
                                yContComp = int(momentComp["m01"] / momentComp["m00"])
                                distance = ((abs(xCont-xContComp))**2.0+(abs(yCont-yContComp))**2.0)**0.5
                                if (distance < 30): #TODO check (this is d1)
                                        eachPlantContours[contPlantDict[str(j)]].append(contCompare)
                                        eachPlantIndex.append(i)
                                        contPlantDict[str(i)] = len(eachPlantContours)-1
                        i += 1
                j += 1
        croppedIndex += 1

'''rowXs = []
rowYs = []
for contList in eachPlantContours:
        print(len(contList))
        minX = 1000000;
        maxX = -1
        minY = 1000000;
        maxY = -1
        for cont in contList:
                moment = cv2.moments(cont)
                x = int(moment["m10"] / moment["m00"])
                y = int(moment["m01"] / moment["m00"])
                minX = min(x, minX)
                maxX = max(x, maxX)
                minY = min(y, minY)
                maxY = max(y, maxY)
        rowXs.append((minX+maxX)/2.0)
        rowYs.append((minY+maxY)/2.0)'''


# TODO Maybe do the simplify (idk quite how, and it seems to work alright otherwise, so it's probably nbd)


# Do regression on plant coordinates
rowLinear = LinearRegression().fit(np.array(rowXs).reshape((-1,1)), np.array(rowYs))
r2 = rowLinear.score(np.array(rowXs).reshape((-1,1)), np.array(rowYs))
intercept = rowLinear.intercept_
final = rowLinear.intercept_ + 640 * rowLinear.coef_
print("R^2: "+str(r2))

# Show images
cv2.line(image, (0, potentialRow), (640, potentialRow), (255, 255, 0))
cv2.line(image, (0, int(initialIntercept)), (640, int(initialReg[0])), (0,255,255))
cv2.line(image, (0, int(intercept)), (640, int(final[0])), (255,255,255))

i = 0
for x in rowXs:
        cv2.circle(image, (x, rowYs[i]), 3, (0,255,0))
        i += 1
        
#cv2.drawContours(image, contours, -1, (0,0,0), 1)
#cv2.drawContours(image, inRangeToPotential, -1, (255,0,255), 1)
#for cont in eachPlantContours:
#        h,s,l = random.random(), 0.5 + random.random()/2.0, 0.4 + random.random()/5.0
#        r,g,b = [int(256*i) for i in colorsys.hls_to_rgb(h,l,s)]
#        g = g/10
#        cv2.drawContours(image, cont, -1, (b,g,r), 1)

        
cv2.imshow("linear combination", linearCombine)
cv2.imshow("threshold", threshImage)
cv2.imshow("filtered", filtered)
cv2.imshow("sobel", sobel)
cv2.imshow("original", image)


cv2.waitKey(0)

