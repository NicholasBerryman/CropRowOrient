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

outFile = open('greenAccumVertical.csv', 'w')
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
                windowOffset = 0

                # Row counting for approximate potential centre line
                cols = np.hsplit(filtered, imageWidth)
                count = 0
                potentialRow = -1
                i = 0
                for row in cols:
                        if (i > imageWidth/2 - windowWidth + windowOffset and i < imageWidth/2 + windowWidth + windowOffset):
                                newCount = cv2.countNonZero(row)
                                if (newCount > count):
                                        count = newCount
                                        potentialRow = i 
                        i += 1



                # Draw Images
                cv2.line(image, (int(potentialRow), 0), (int(potentialRow),imageHeight), (0,255,255), 3)
                i = 0
                        
                cv2.imshow("linear combination", linearCombine)
                cv2.imshow("threshold", threshImage)
                cv2.imshow("filtered", filtered)
                cv2.imshow("original", image)
                cv2.waitKey(0)

                angle = 90
                midpoint = potentialRow
                realAngle = path.split("_")[1]
                ID = path.split("_")[0]
                outFile.write(str(ID)+','+str(realAngle)+','+str(angle)+','+str(midpoint)+'\n')
outFile.close()

