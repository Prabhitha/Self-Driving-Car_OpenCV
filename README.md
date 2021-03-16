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
plt.imshow(image)
plt.title("Test Image")
plt.show
```
![image](https://user-images.githubusercontent.com/26111880/111380253-180db780-86ca-11eb-9d15-1dcf0f095e0e.png)
![GitHub Logo](/images/logo.png)
Format: ![Alt Text](url)



