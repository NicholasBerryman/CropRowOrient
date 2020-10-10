import math
import os
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

outFile = open('horiStripsVertical.csv', 'w')
outFile.write("ID,Real Angle,Estimated Angle,Line Midpoint\n")
for path in os.listdir():
        if '.jpg' in path:
                imgName = path#input("File Name: ")
                image = cv2.imread(imgName)


                # Rotate these images because they are portrait
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

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

                #Define image dimensions
                imageWidth = 480
                imageHeight = 640


                # Defining our window
                windowWidth = 100
                windowOffset = -25
                plantWindowWidth = 150


                #Split image into segments
                nRows = 15
                yOffset = imageHeight/nRows
                imageParts = list()
                for i in range(0,nRows):
                        imageParts.append(filtered[i*(int)(imageHeight/nRows):i*(int)(imageHeight/nRows)+(int)(imageHeight/nRows),(int)(imageWidth/2 - windowWidth + windowOffset):imageWidth])
                i = 0
                for c in imageParts:
                        i+=1
                        #cv2.imshow("cropped"+str(i),c)



                # Column counting for approximate potential centre line
                cols = np.hsplit(filtered, imageWidth)
                count = 0
                potentialRow = -1
                i = 0
                for row in cols:
                        if i <  windowWidth*2:
                                newCount = cv2.countNonZero(row)
                                if (newCount > count):
                                        count = newCount
                                        potentialRow = i 
                        i += 1

                # Contour detection
                i = 0
                rowXs = list()
                rowYs = list()
                allContours = list()
                for part in imageParts:
                        contours, heirarchy = cv2.findContours(part, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                        largestSize = 0
                        largestCont = []
                        for contour in contours:
                                cArea = cv2.contourArea(contour)
                                cx, cy = getContCentre(contour)
                                cx += (int)(imageWidth/2 - windowWidth + windowOffset)
                                if (abs(potentialRow - cx) < plantWindowWidth):
                                        if (cArea > largestSize):
                                                largestSize = cArea
                                                largestCont = contour
                        if (largestSize > 0):
                                largestCont += [(int)(imageWidth/2 - windowWidth + windowOffset),int(yOffset * i)]
                                allContours.append(largestCont)
                                cx, cy = getContCentre(largestCont)
                                rowXs.append(cx)
                                rowYs.append(cy)
                        i+=1


                # Do regression on plant coordinates
                rowLinear = LinearRegression().fit(np.array(rowYs).reshape((-1,1)), np.array(rowXs))
                r2 = rowLinear.score(np.array(rowXs).reshape((-1,1)), np.array(rowYs))
                intercept = rowLinear.intercept_
                final = rowLinear.intercept_ + imageWidth * rowLinear.coef_

                # Draw Images
                cv2.line(image, (int(intercept), 0), (int(final[0]), imageHeight), (0,255,255), 3)

                cv2.drawContours(image, allContours, -1, (255,255,255), 1)
                i = 0
                for x in rowXs:
                        cv2.circle(image, (x, rowYs[i]), 3, (0,255,0), 1)
                        i += 1
                cv2.imshow("linear combination", linearCombine)
                cv2.imshow("threshold", threshImage)
                cv2.imshow("filtered", filtered)
                cv2.imshow("original", image)
                cv2.waitKey(0)

                rise = final-intercept
                run = imageHeight
                angle = ((3.1415/2.0)-math.atan(rise/run))/3.1415*180
                midpoint = ((intercept+rise/2.0)[0], imageHeight/2.0)
                realAngle = path.split("_")[1]
                ID = path.split("_")[0]
                outFile.write(str(ID)+','+str(realAngle)+','+str(angle)+','+str(midpoint[0])+'\n')
outFile.close()
                

