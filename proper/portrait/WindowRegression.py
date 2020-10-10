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

outFile = open('windowRegVertical.csv', 'w')
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

                # Row counting for approximate potential centre line
                rows = np.vsplit(filtered, imageHeight)
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
                xOffset = imageWidth/nCols
                imageParts = list()
                for i in range(0,nCols):
                        imageParts.append(filtered[0:imageHeight, i*(int)(imageWidth/nCols):i*(int)(imageWidth/nCols)+(int)(imageWidth/nCols)])
                i = 0
                for c in imageParts:
                        i+=1
                        cv2.imshow("cropped"+str(i),c)

                # Defining our window
                windowWidth = 75
                windowHeight = 200
                windowOffsetY = 0
                windowOffsetX = 50
                windowMinX = int(imageWidth/2 - windowWidth - windowOffsetX)
                windowMaxX = int(imageWidth/2 + windowWidth - windowOffsetX)
                windowMinY = int(imageHeight/2 - windowHeight - windowOffsetY)
                windowMaxY = int(imageHeight/2 + windowHeight - windowOffsetY)



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
                rowLinear = LinearRegression().fit(np.array(rowYs).reshape((-1,1)), np.array(rowXs))
                r2 = rowLinear.score(np.array(rowXs).reshape((-1,1)), np.array(rowYs))
                intercept = rowLinear.intercept_
                final = rowLinear.intercept_ + imageWidth * rowLinear.coef_
                print("R^2: "+str(r2))

                # Draw Images
                cv2.line(image, (int(intercept), 0), (int(final[0]), imageHeight), (0,255,255), 3)
                cv2.rectangle(image, (windowMinX, windowMinY), (windowMaxX, windowMaxY), (255,0,0), 1)
                i = 0
                for x in rowXs:
                        cv2.circle(image, (x, rowYs[i]), 3, (0,255,0))
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
                

