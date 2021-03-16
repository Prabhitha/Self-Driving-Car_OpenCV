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

### Gray Image
![image](https://user-images.githubusercontent.com/26111880/111389247-a7b96300-86d6-11eb-94ff-bef19bf6cf44.png)

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
![image](https://user-images.githubusercontent.com/26111880/111389329-cae41280-86d6-11eb-955c-397a6071c35f.png)




