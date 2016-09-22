import itertools as it
import pickle
from random import sample, randint, random
from time import time
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import operator
import math
import glob

from PIL import Image as PILImage
import sys
import Image
import os
from random import shuffle
import copy
#import scipy.misc

load_dir = '/media/shehroze/sheep/deathmatch_slam_new/'
save_dir = '/media/shehroze/sheep/deathmatch_slam_new/'

total_images = 991

def convertToGrayScale(img, height=84,width=84):
    img = img.astype(np.float32) 
    img = cv2.resize(img, (height, width))
    return img

def convert(img):
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    rgb[..., 0] = img[:,:,2]
    rgb[..., 1] = img[:,:,1]
    rgb[..., 2] = img[:,:,0]
    return rgb

count = 0

with open(load_dir + 'depth.txt') as f:
	frameCount = -1
	for line in f:

		
		line_components = line.split(" ")
		filename = line_components[1].replace("depth/", "")	
		filename = filename.replace("\n","")
		print('Converting file ' + filename)
		depth_file_name = 'depth/' + filename
		rgb_file_name = 'rgb/' + filename


		depth_file = cv2.imread(load_dir + depth_file_name, cv2.CV_LOAD_IMAGE_UNCHANGED)
		rgb_file   = cv2.imread(load_dir + rgb_file_name)	


		

		print(depth_file.shape)
		print(rgb_file.shape)
		
	
		new_depth_file = cv2.resize(depth_file,(160,120))
		new_rgb_file   = np.zeros((120,160,3))

		new_rgb_file[:,:,0], new_rgb_file[:,:,1], new_rgb_file[:,:,2] = cv2.resize(rgb_file[:,:,0], (160,120)), cv2.resize(rgb_file[:,:,1], (160,120)), cv2.resize(rgb_file[:,:,2], (160,120))

		cv2.imwrite(save_dir + 'depth_small_size/' + filename , new_depth_file)		
		cv2.imwrite(save_dir + 'rgb_small_size/' + filename, new_rgb_file)		

