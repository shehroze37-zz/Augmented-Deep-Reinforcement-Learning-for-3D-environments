import itertools as it
import pickle
from random import sample, randint, random
from time import time
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import operator
import math

from PIL import Image as PILImage
import sys
import Image
import os
from random import shuffle
import copy

load_dir = 'map-construction-data/'
save_dir = 'Frames/'

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

timestamp = True
if timestamp == True:
	index = 0
	with open(load_dir + 'rgb.txt') as f:
		for line in f:
			components = line.split(' ')


			print('Working on file ======>  ' + str(components[1].strip('\n').strip('.png')))

			#get the image
			image_file_components = components[1].strip('\n').split('/')
			image_file = load_dir + 'original/' +  image_file_components[1]


			
			im = cv2.imread(image_file)
			height,width,depth = im.shape
	
			#save rgb image to ppm format
			modified_image = convert(im)
			output = Image.fromarray(modified_image)
			output.save(save_dir + str(index).zfill(4) + '.ppm')

			#generate pgm for mask
			original = True
			masked_data = None
			depth_map_file = image_file_components[1].split('.')
			depth_map_file = depth_map_file[0] + '.' + depth_map_file[1]
			if original == True:
				data = pickle.load(open(load_dir + 'depth_map/' + str(depth_map_file) + '.txt','rb'))
			
			data_conv = data.astype(np.uint16)
			cv2.imwrite(save_dir + str(index).zfill(4) + '.pgm', data_conv)

			index += 1

	sys.exit()

for i in range(total_images):

	print('Working on file ======>  ' + str(i))

	#get the image
	image_file = load_dir + 'original/rgb_' + str(i) + '.jpg'
	im = cv2.imread(image_file)
	height,width,depth = im.shape
	
	#save rgb image to ppm format
	modified_image = convert(im)
	output = Image.fromarray(modified_image)
	output.save(save_dir + str(i).zfill(4) + '.ppm')


	#generate pgm for mask
	original = True
	masked_data = None
	if original == True:
		data = pickle.load(open(load_dir + 'depth_map/depth_map_original_' + str(i) + '.txt','rb'))
	else:
		data = pickle.load(open(load_dir + 'new_masked/masked_depth_map_' + str(i) + '.txt','rb'))
	
	data_conv = data.astype(np.uint16)
	cv2.imwrite(save_dir + str(i).zfill(4) + '.pgm', data_conv)
