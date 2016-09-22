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



load_dir = '/home/shehroze/Documents/VOCdevkit2/VOC2012/'
save_dir = ''

total_images = 3156

#sample_image = '/home/shehroze/Documents/VOCdevkit2/VOC2012/JPEGImages/572_bounding.jpg'
color_dic_original = {'h1' : [120,102,210], 'h2' : [75,240,78] ,'h3': [11,163,98], 'h4' : [45,115,102],'h5' : [28,30,60],'a1' : [75,65,240],'a2' : [81,41,61],'a3' : [63,67,99],'a4' : [97,193,163],'a5' : [211,213,217],'a6':[241,233,133],'a7' : [137,199,61],'a8' : [55,155,155],'a9' : [22,122,222],'e1' : [19,129,229],'e2' : [61,97,261],'e3' : [142,149,147],'e4' : [77,177,197],'e5' : [88,188,208],'e6' : [43,143,173],'e7' : [57,150,207],'e8' : [59,79,179],'c1' : [18,81,181],'c2' : [71,27,207],'c3' : [93,53,103],'c4' : [203,103,95],'c5' : [64,104,204]}

color_dic_list = [ 'h1' , 'h2'  ,'h3', 'h4' ,'h5' ,'a1' ,'a2' ,'a3' ,'a4' ,'a5' ,'a6' ,'a7' ,'a8' ,'a9' ,'e1' ,'e2' ,'e3' ,'e4' ,'e5' ,'e6' ,'e7' ,'e8' ,'c1' ,'c2' ,'c3' ,'c4' ,'c5' ]
#color_dic_list                  = [ 'h1',  'h2', 'h3', 'h4', 'h5','e8', 'e4', 'e6', 'e7', 'c5',  'a2', 'a1', 'c1', 'a9', 'e3', 'a7', 'a4', 'a8', 'a6', 'a5', 'e1', 'e2', 'c2', 'e5', 'c4']


pallete = []
final_classes = []

if len(color_dic_original) != len(color_dic_list):
	print('ERROR')
	sys.exit()

color_dic = {}
enemies_class = ['e1' ,'e2' ,'e3' ,'e4' ,'e5' ,'e6' ,'e7']
ammo_classes  = ['a1' ,'a2' ,'a4'  ,'a6' ,'a8' ,'a9']

for i in range(len(color_dic_list)):

	object_class = color_dic_list[i]
	color = 0

	if object_class == 'h1' or object_class == 'h2' or object_class == 'h3' or object_class == 'h4' or object_class == 'h5':
		color = 0		
	elif object_class in enemies_class:
		color = 1
	elif object_class == 'a5' or object_class == 'a7':
		#high grade weapons
		color = 2
	elif object_class == 'a3' or object_class == 'e8':
		#high grade ammo
		color = 3
	elif object_class in ammo_classes:
		color = 4
	elif object_class == 'c4' or object_class == 'c5':
		color = 5
	elif object_class == 'c1' or object_class == 'c2' or object_class == 'c3':
		color = 6
	else:
		print('ERROR defining colors ')
		sys.exit()

	color_dic[color_dic_list[i]] = [color,color,color]
	pallete.append(color)
	pallete.append(color)
	pallete.append(color)

for i in range(total_images):

	#get the image
	image_file = load_dir + 'JPEGImages/' + str(i) + '.jpg'
	im = cv2.imread(image_file)
	height,width,depth = im.shape
	rectangle_img = np.zeros((height,width), np.uint8)
	rectangle_img.fill(255)

	#get the xml file 
	xml_file = load_dir + 'Annotations/' + str(i) + '.xml'
	tree = ET.parse(xml_file)
	root = tree.getroot()
	objs = tree.findall('object')

	box_size = {}
	box_dims = {}
	box_class = {}

	current_index = 0
	for ix, obj in enumerate(objs):

		name = obj.find('name').text
		bbox = obj.find('bndbox')
		
		for visit_element in bbox.iter('bndbox'):
	
			x1 = int(math.floor(float(bbox.find('xmin').text)))
			x2 = int(math.ceil(float(bbox.find('xmax').text)))
			y1 = int(math.floor(float(bbox.find('ymin').text)))
			y2 = int(math.ceil(float(bbox.find('ymax').text)))
		
			if x1 < 10 or x2 > 630 or y1 < 0 or y2 > 480:
				continue
			elif (x2 - x1) <= 10 and (y2 - y1) <= 10 :
				continue
			else:
				box_size[current_index] = (x2 - x1) * (y2 - y1)
				box_dims[current_index] = [x1,x2,y1,y2]
				box_class[current_index] = name
				current_index += 1
	
		if name not in final_classes:
			final_classes.append(name)

	sorted_size = sorted(box_size.items(), key=operator.itemgetter(1))
	'''for j in range(len(sorted_size)):
		key = sorted_size[j][0]
	
		x1 = box_dims[key][0]
		x2 = box_dims[key][1]
		y1 = box_dims[key][2]
		y2 = box_dims[key][3]

		cv2.rectangle(rectangle_img, (x1, y1), (x2,y2), 280, -1)

		
	masked_data = cv2.bitwise_and(im, im, mask=rectangle_img)'''
	masked_data = rectangle_img

	for j in range(len(sorted_size)):
		key = sorted_size[j][0]
	
		x1 = box_dims[key][0]
		x2 = box_dims[key][1]
		y1 = box_dims[key][2]
		y2 = box_dims[key][3]

		color_list = color_dic[box_class[key]]
		cv2.rectangle(masked_data,(x1, y1), (x2,y2),(color_list[0],color_list[1],color_list[2]),-1)

	#cv2.imshow("masked", masked_data)
	print('Generating image ' + str(i))
	#cv2.imwrite(load_dir + 'segmentationclass/' + str(i) + '.png', masked_data)
	output=Image.fromarray(masked_data)
	output.save(load_dir + 'segmentationclass/' + str(i) + '.png')

	#output_im = PILImage.fromarray(masked_data)
	#output_im.putpalette(pallete)
	#cv2.imwrite(load_dir + 'segmentationclass/' + str(i) + '_with_palette.png', output_im)
	#cv2.waitKey(0)

print(final_classes)
print('Total classes = ' + str(len(final_classes)))
