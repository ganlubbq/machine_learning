# -*- coding: utf-8 -*-
"""
find_phone.py
Usage: python find_phone.py /path/to/test_image.jpg

Uses Haar/LBP cascades generated in train_phone_finder.py to detect objects. 

Assumptions:
    - OpenCV is installed and is in the path.
	- The object is roughly the same size in the test image as the training images.

Author: Aaron Penne
Date: October 2017
References:
	- Viola/Jones paper - https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-cvpr-01.pdf 
	- OpenCV docs - https://docs.opencv.org/2.4/doc/user_guide/ug_traincascade.html
"""

import sys
import cv2

###############################################################################
#%% Helper functions
###############################################################################
def norm_coords(x_in, y_in, img):
    try:
        rows = len(img)
        cols = len(img[0])
    except:
        cols, rows = img.size
    if x_in<=1 and y_in<=1:
        y = y_in*rows
        x = x_in*cols
    else:
        y = y_in/(rows*1.0)
        x = x_in/(cols*1.0)
    return x, y

###############################################################################
#%% Main
###############################################################################

# Verifies user input matches desired input.
if not len(sys.argv)==2:
	sys.exit('Usage: python train_phone_finder.py /path/to/images')

# Reads in image and trained model.
img = cv2.imread(sys.argv[1])
cascade = cv2.CascadeClassifier('./output/cascade.xml')

# Converts image to grayscale.
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detects objects matching trained model. We are able to limit the rectangle
# size range due to the size assumption.
phone = cascade.detectMultiScale(img_gray, 
                                 scaleFactor=1.3, 
                                 minNeighbors=5, 
                                 minSize=(30,30), 
                                 maxSize=(100,100))

# Pulls out first detected phone. This method is naive, and should be improved 
# in later versions with weights or grouping rectangles.
try:
	x = phone[0][0]
	y = phone[0][1]
	w = phone[0][2]
	h = phone[0][3]
except:
	sys.exit('No object detected')
	
# Finds center of rectangle, if all went well this is also the center of the object.
x_center = x+(w/2)
y_center = y+(h/2)
x_norm, y_norm = norm_coords(x_center, y_center, img_gray)

print('{0:.3f} {1:.3f}'.format(x_norm, y_norm))

# Uncomment if you want to display the annotated image.	
img = cv2.rectangle(img, 
                    (x, y), 
                    (x+w, y+h), 
                    (0, 102, 255), 
                    2)				
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
