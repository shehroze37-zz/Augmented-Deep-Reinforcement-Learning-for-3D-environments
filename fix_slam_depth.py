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

load_dir = '/media/shehroze/sheep/triangular-health/'
save_dir = '/media/shehroze/sheep/triangular-health/depth/'

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
for filename in glob.iglob(load_dir + ' old_depth/*.txt'):

	print('Working on file ==> ' + str(filename))

	#get the image
	image_file = load_dir + 'depth/' + str(filename) + '.txt'
	im = pickle.load(open(filename,'rb'))

	#save rgb image to ppm format

	modified_image = (im / float(32)) * 5000
	modified_image = modified_image.astype(np.uint16)

	name = filename.split('/')
	name_components = name[len(name) - 1].split('.')
	#scipy.misc.imsave(save_dir + name_components[0] + '.' + str(name_components[1]) +  '.png', modified_image)'''
	


	#3channel png depth
	#new_depth = np.zeros((480, 640, 1), dtype=np.uint16)
	#new_depth[:,:,0] = modified_image
	#new_depth[:,:,1] = modified_image
	#new_depth[:,:,2] = modified_image

	#scipy.misc.imsave(save_dir + name_components[0] + '.' + str(name_components[1]) +  '.png', modified_image)
	cv2.imwrite(save_dir + name_components[0] + '.' + str(name_components[1]) +  '.png', modified_image)
	count += 1


print('Total files converted = ' + str(count))
