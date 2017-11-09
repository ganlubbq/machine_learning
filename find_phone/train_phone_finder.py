# -*- coding: utf-8 -*-
"""
train_phone_finder.py
Usage: python train_phone_finder.py /path/to/images_to_train

Uses Haar/LBP cascades to detect objects. Haar cascades use positive and
negative image datasets to train the detector. Both the positive and negative
sets are generated from a single set of images. Negatives are created by covering
the object with a duplicated adjacent portion of the image.

Assumptions:
    - OpenCV is installed and is in the path.
    - The object is roughly the same size in all the images (for automated annotation).
    - There is only one object per positive image.

Author: Aaron Penne
Date: October 2017
References:
	- Viola/Jones paper - https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-cvpr-01.pdf 
	- OpenCV docs - https://docs.opencv.org/2.4/doc/user_guide/ug_traincascade.html
"""

import sys
import os
import csv
import glob
import shutil
from PIL import Image

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

# Verifies user input matches desired input
if len(sys.argv)==2:
	if os.path.isdir(sys.argv[1]):
		dir_train = sys.argv[1]	
	else:
		sys.exit('Usage: python train_phone_finder.py /path/to/images')
else:
	sys.exit('Usage: python train_phone_finder.py /path/to/images')
    
file_labels = 'labels.txt'
file_pos = 'positive.txt'
file_neg = 'negative.txt'
file_vec = 'out.vec'
text_gray = '_gray.jpg'
text_crop = '_crop.jpg'
text_filled = '_filled.jpg'

# Cleanup files from previous runs
if os.path.exists(file_pos):
    os.remove(file_pos)
if os.path.exists(file_neg):
    os.remove(file_neg)

# Makes dir for cropped positive images
dir_pos = os.path.join(dir_train, 'positive')
if not os.path.exists(dir_pos):
    os.makedirs(dir_pos)

# Makes dir for negative images
dir_neg = os.path.join(dir_train, 'negative')
if not os.path.exists(dir_neg):
    os.makedirs(dir_neg)
    
# Reads in file with coordinates
try:
    with open(os.path.join(dir_train, file_labels), 'rb') as file_in:
        reader = csv.reader(file_in, delimiter=' ')
        labels = [[row[0], float(row[1]), float(row[2])] for row in reader]
except IOError as error:
    sys.exit(error)
    
# Generates positives and negatives
# Opens all images, converts to grayscale, crops box around phone, saves to corresponding dir
for row in labels:
    try:
        img = Image.open(os.path.join(dir_train, row[0]))
    except IOError as error:
        print 'WARNING: ', error
        continue
  
    # Generates grayscale positive images
    img = img.convert('L')
    img_path = os.path.join(dir_pos, row[0][:-4]+text_gray)
    img.save(img_path)

    # Annotates images given center location from label text file and expanded rectangular area
    radius = 30  # pixels - defines size of rectangle
    x, y = norm_coords(row[1], row[2], img)
    box = (x-radius,
           y-radius,
           x+radius,
           y+radius)
    
    # Writes text file with positive images annotated as opencv
    with open(file_pos, 'a') as file_out:
        file_out.write('{} {} {} {} {} {}\n'.format(img_path, 1, int(box[0]), int(box[1]), radius*2, radius*2))
    
    # Generates negative images
    # Covers the phone with another part of the image, either from the immediate left or right of phone
    img_neg = img
    if box[0]>radius*2:
        filler = (box[0]-radius*2,
                  box[1],
                  box[2]-radius*2,
                  box[3])
    else:
        filler = (box[0]+radius*2,
                  box[1],
                  box[2]+radius*2,
                  box[3])   
    box = [int(ii) for ii in box]  # Converts to int for paste operation
    img_filler = img_neg.crop(filler)
    img_neg.paste(img_filler, (box[0], box[1]))
    img_path = os.path.join(dir_neg, row[0][:-4]+text_filled)
    img_neg.save(img_path)
    neg_list = glob.glob(os.path.join(dir_neg, '*jpg'))
    with open(file_neg, 'w') as file_out:
        for row in neg_list:
            file_out.write('{}\n'.format(row))

# Creates vector file of positive images and corresponding annotations to be used for training
sys_string = 'opencv_createsamples ' \
             '-info ' + file_pos + ' ' \
             '-vec ' + file_vec + ' ' \
             '-bg ' + file_neg + ' ' \
             '-num ' + str(len(labels)) + ' ' \
             '-w ' + str(int(radius*2)) + ' ' \
             '-h ' + str(int(radius*2))
os.system(sys_string)
 
# opencv_traincascade uses previous parameters in output dir, this is undesirable so the output dir is backed up, deleted, then created again
output_dir = 'output'
output_path = os.path.join('.',output_dir)
if os.path.isdir(output_path):
    if os.path.isdir(output_path+'_BAK'):
        shutil.rmtree(output_path+'_BAK')    
    shutil.copytree(output_path,output_path+'_BAK')
    shutil.rmtree(output_path)
os.makedirs(output_path)

# Trains cascade using positive and negative samples derived from opencv_createsamples.
# Function throws out images that are too similar, need to reduce count of positives by some amount
sys_string = 'opencv_traincascade ' \
             '-featureType LBP ' \
             '-minHitRate 0.996 ' \
             '-maxFalseAlarmRate 0.5 ' \
             '-data ' + output_dir + ' ' \
             '-vec ' + file_vec + ' ' \
             '-bg ' + file_neg + ' ' \
             '-numPos ' + str(int(len(labels)*0.85)) + ' '  \
             '-numNeg ' + str(len(neg_list)) + ' ' \
             '-numStages 21 ' \
             '-acceptanceRatioBreakValue 0.00001 ' \
             '-w ' + str(int(radius*2)) + ' ' \
             '-h ' + str(int(radius*2))
os.system(sys_string)
