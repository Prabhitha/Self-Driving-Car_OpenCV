import cv2
import numpy as np
import matplotlib.pyplot as plt

def canny(image):
    gray = cv2.cvtColor(lane_image,cv2.COLOR_RGB2GRAY) # Converting a color image to gray image to reduce computational complexity
    blur = cv2.GaussianBlur(gray,(5,5),0) # To reduce noise and smoothen the image
    canny = cv2.Canny(blur,50,150) # To detect the edges of the image using canny fn
    return canny

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    line_image = cv2.cvtColor(line_image,cv2.COLOR_GRAY2RGB)
    if lines is not None: # to check if any line is detected, lines- 3-d array
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            # line- draws a line segment connecting 2 points, color of the line, line density
            cv2.line(line_image, (x1,y1), (x2,y2), (255,0,0), 10)
    return line_image

def region_of_interest(image): # Detect the lanes(enclosed region):
    height = image.shape[0] # To get the height of the image
    polygons = np.array([[(200,height),(1100,height),(550,250)]]) # Providing the 3 edge values of the triangle
    mask = np.zeros_like(image) # Create an array of zeros with the shape of the image (black image)
    # The mask will be an image which is completely black
    cv2.fillPoly(mask,polygons,255) # Fill the mask with our triangle(with color white)
    masked_image = cv2.bitwise_and(image,mask) # shows only the triangle in white color
    return masked_image # Modified mask

image = cv2.imread('test_image.jpg')
lane_image = np.copy(image)
canny = canny(lane_image)
cropped_image = region_of_interest(canny)
# HoughLinesP
# 1st param- Image where we want to look at the lines
# 2nd and 3rd param - Specify the resolution of half accumulator array (2 dim array/grid which has the max votes of intersection
# It specify the size of the bins, 2 pixels and 1-degree precision (pi/180)
# 4th param- Threshold - minimum no of votes needed to accept a candidate line.
# 5th param - placeholder array (Empty array)
# 6th param -Length of the line in pixels that we accept into the output
# 7th param -max dist in pixels between segmented lines which we allow to be connected into single line instead of them broken up
lines = cv2.HoughLinesP(cropped_image,2, np.pi/180,100,np.array([]),minLineLength = 40,maxLineGap = 5)
line_image = display_lines(cropped_image, lines)
combo_image = cv2.addWeighted(lane_image,0.8,line_image,1,1) # To blend the line with the original image, pixel intensity
#cv2.imshow('result',canny)
cv2.imshow('result',combo_image)
cv2.waitKey(0)
#plt.imshow(canny)
#plt.show() # Image will have x-axis and y-axis values
