## SELF DRIVING CAR - OpenCV
### We are going to detect lanes in the roads using OpenCv library.
OpenCV (Open Source Computer Vision Library) is a library of programming functions mainly aimed at real-time computer vision.<br>
It mainly focuses on image processing, video capture and analysis including features like face detection and object detection.<br>
To install OpenCV- pip install opencv-python

```
# Importing the libraries
import cv2 # Dataset
import numpy as np
import matplotlib.pyplot as plt
```
```
# load the image
path = r'C:\Users\Prabhitha Nagarajan\Desktop\finding-lanes\test_image.jpg'
image = cv2.imread(path)
  
# Displaying the image 
plt.axis("off")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BG))
```
![image](https://user-images.githubusercontent.com/26111880/111388623-a0de2080-86d5-11eb-8889-609d1d08346f.png)

We are going to detect the edges of the image.<br>
Edge Detection - Identifying sharp changes in intensity in adjacent pixels <br>
1. Convert RGB image to grey scale to reduce the computational complexity
2. Noises can create false edges, so we need to reduce it. We are going to smoothen the image using Gausssian filter
3. Using Canny fn, we are going to detect the edges.

```
def canny(image):
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY) 
    blur = cv2.GaussianBlur(gray,(5,5),0) 
    canny = cv2.Canny(blur,50,150) 
    return canny

lane_image = np.copy(image)
canny_image = canny(lane_image)
plt.imshow(cv2.cvtColor(canny_image, cv2.COLOR_BGR2RGB))
```
![image](https://user-images.githubusercontent.com/26111880/111389247-a7b96300-86d6-11eb-94ff-bef19bf6cf44.png)![image](https://user-images.githubusercontent.com/26111880/111389329-cae41280-86d6-11eb-955c-397a6071c35f.png)

Create a black image with the shape of the original image (Array of zeros) and fill the mask image with our lane in white color<br>
```
def region_of_interest(image): 
    height = image.shape[0] 
    polygons = np.array([[(200,height),(1100,height),(550,250)]])
    mask = np.zeros_like(image) 
    cv2.fillPoly(mask,polygons,255) 
    masked_image = cv2.bitwise_and(image,mask)
    return masked_image 

cropped_image = region_of_interest(canny_image)
plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
```
![image](https://user-images.githubusercontent.com/26111880/111393456-d0ddf180-86de-11eb-8b9d-a46eb0b36f2f.png)

### Hough transform

The Hough transform is a feature extraction technique used in image analysis, computer vision, and digital image processing.The purpose of the technique is to find imperfect instances of objects within a certain class of shapes by a voting procedure. This voting procedure is carried out in a parameter space, from which object candidates are obtained as local maxima in a so-called accumulator space that is explicitly constructed by the algorithm for computing the Hough transform.

HoughLinesP<br>

1st param- Image where we want to look the lines<br>
2nd and 3rd param - Specify the resolution of half accumulator array (2 dim array/grid which has the max votes of intersection. It specify the size of the bins, 2 pixels and 1-degree precision (pi/180)<br>
4th param- Threshold - minimum no of votes needed to accept a candidate line<br>
5th param - placeholder array (Empty array)<br>
6th param -Length of the line in pixels that we accept into the output<br>
7th param -max dist in pixels between segmented lines which we allow to be connected into single line instead of them to be   broken up

```
lines = cv2.HoughLinesP(cropped_image,2, np.pi/180,100,np.array([]),minLineLength = 40,maxLineGap = 5)

# Displaying the lines in a black image
def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None: # to check if any line is detected, lines- 3-d array
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            # line- draws a line segment connecting 2 points, color of the line, line density
            cv2.line(line_image, (x1,y1), (x2,y2), (255,0,0), 10)
    return line_image

line_image = display_lines(lane_image, lines)
plt.imshow(cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB))
```
![image](https://user-images.githubusercontent.com/26111880/111393506-e521ee80-86de-11eb-9048-397f9b033d4e.png)

To blend the line with the original image
```
combo_image = cv2.addWeighted(lane_image,0.8,line_image,1,1) 
plt.imshow(cv2.cvtColor(combo_image, cv2.COLOR_BGR2RGB))
```
![image](https://user-images.githubusercontent.com/26111880/111393532-f0751a00-86de-11eb-805b-bed046ceca29.png)

Changing multiple lines in the lane to single line - Optimization<br>
Get the slope and y-intercept of all the lines, average them and create a new line.<br>
polyfit- fits a 1st degree polynomial(y= mx+b)

```
def make_coordinates(image,line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0] # Image shape: (704,1279,3), y1= 704
    y2 = int(y1*(3/5)) # y2= 422.4
    x1 = int((y1-intercept)/slope)
    x2 = int((y2-intercept)/slope)
    return np.array([x1,y1,x2,y2])

def average_slope_intercept(image,lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2),(y1,y2), 1) 
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope,intercept))
        else:
            right_fit.append((slope,intercept))
    left_fit_average = np.average(left_fit, axis=0) # Calculate average slope and y-intercept vertically
    right_fit_average = np.average(right_fit, axis=0)
    left_line = make_coordinates(image,left_fit_average) # To get the coordinates to draw the line
    right_line = make_coordinates(image,right_fit_average)
    return np.array([left_line,right_line])

average_lines = average_slope_intercept(lane_image, lines) 
line_image = display_lines(lane_image, average_lines)
plt.imshow(cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB))
```
![image](https://user-images.githubusercontent.com/26111880/111393586-0aaef800-86df-11eb-8ed2-8245aa8afd78.png)
 
To blend the line with the original image 
```
combo_image = cv2.addWeighted(lane_image,0.8,line_image,1,1) 
plt.imshow(cv2.cvtColor(combo_image, cv2.COLOR_BGR2RGB))
 ```
![image](https://user-images.githubusercontent.com/26111880/111393651-27e3c680-86df-11eb-8c0c-a3685858a226.png)

### Capture Lanes from videos
We are going to decode every video frame -> It will result in multiple images, we will then apply the above lane detection funtion on top of that

```
cap = cv2.VideoCapture("test2.mp4")
while(cap.isOpened()):
    _, frame = cap.read() # To decode every video frame
    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_image,2, np.pi/180,100,np.array([]),minLineLength = 40,maxLineGap = 5)
    average_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, average_lines)
    combo_image = cv2.addWeighted(frame,0.8,line_image,1,1)
    cv2.imshow('result',combo_image)
    #cv2.waitKey(1) - wait 1ms in between frames
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
```




