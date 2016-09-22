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

load_dir = '/home/shehroze/Documents/VOCdevkit2/VOC2012-old/'
save_dir = '/home/shehroze/Documents/VOCdevkit3/VOC2012/'

total_images = 3156


color_dic_original = {'h1' : [120,102,210], 'h2' : [75,240,78] ,'h3': [11,163,98], 'h4' : [45,115,102],'h5' : [28,30,60],'a1' : [75,65,240],'a2' : [81,41,61],'a3' : [63,67,99],'a4' : [97,193,163],'a5' : [211,213,217],'a6':[241,233,133],'a7' : [137,199,61],'a8' : [55,155,155],'a9' : [22,122,222],'e1' : [19,129,229],'e2' : [61,97,261],'e3' : [142,149,147],'e4' : [77,177,197],'e5' : [88,188,208],'e6' : [43,143,173],'e7' : [57,150,207],'e8' : [59,79,179],'c1' : [18,81,181],'c2' : [71,27,207],'c3' : [93,53,103],'c4' : [203,103,95],'c5' : [64,104,204]}

color_dic_list = [ 'h1' , 'h2'  ,'h3', 'h4' ,'h5' ,'a1' ,'a2' ,'a3' ,'a4' ,'a5' ,'a6' ,'a7' ,'a8' ,'a9' ,'e1' ,'e2' ,'e3' ,'e4' ,'e5' ,'e6' ,'e7' ,'e8' ,'c1' ,'c2' ,'c3' ,'c4' ,'c5' ]
final_object_types = []

final_classes = []

object_classes = ['health', 'monster', 'high_grade_weapons', 'high_grade_ammo', 'other_ammo', 'my_shoots', 'monster_shoots']
objectClassesData = {}
for object_type in object_classes:
	objectClassesData[object_type] = []

class_dic = {}
image_object_map = {}
enemies_class = ['e1' ,'e2' ,'e3' ,'e4' ,'e5' ,'e6' ,'e7', 'e8']
ammo_classes  = ['a1' ,'a2' ,'a5', 'a7','a9']

def convertToGrayScale(img, height=84,width=84):
    img = img.astype(np.float32) 
    img = cv2.resize(img, (height, width))
    return img


for i in range(len(color_dic_list)):

	object_class = color_dic_list[i]
	class_obj = 0

	if object_class == 'h1' or object_class == 'h2' or object_class == 'h3' or object_class == 'h4' or object_class == 'h5':
		class_obj = 'health'		
	elif object_class in enemies_class:
		class_obj = 'monster'
	elif object_class == 'a6' or object_class == 'a8':
		#high grade weapons
		class_obj = 'high_grade_weapons'
	elif object_class == 'a3' or object_class == 'a4' or object_class == 'c1' :
		#high grade ammo
		class_obj = 'high_grade_ammo'
	elif object_class in ammo_classes:
		class_obj = 'other_ammo'
	elif object_class == 'c2' or object_class == 'c3' :
		class_obj = 'my_shoots'
	elif object_class == 'c4' or object_class == 'c5':
		class_obj = 'monster_shoots'
	else:
		print('ERROR defining class ' + str(object_class))
		sys.exit()


	if class_obj == 0:
		print('ERROR, wront class')
		sys.exit()

	class_dic[color_dic_list[i]] = class_obj

def getObjectElementInfo(element):
	elements = element.split(" ")
	return elements[0], int(elements[1])


files_removed = []
def saveImageSetData(objectClasses, objectClassesData):

	directory = save_dir + 'ImageSets/Main/'


	
	if not os.path.isdir(directory):
		os.makedirs(directory)

	global_train = []
	global_val = []	

	for object_types in objectClasses:

		positives = []
		negatives = []

		training_list = []
		testing_list  = []
		complete_list = []

		for j in range(len(objectClassesData[object_types])):
			
			image_no, presence = getObjectElementInfo(objectClassesData[object_types][j])
			if presence == 1:
				if objectClassesData[object_types][j] not in positives:
					positives.append(objectClassesData[object_types][j])
			elif presence == -1:
				if objectClassesData[object_types][j] not in negatives:
					negatives.append(objectClassesData[object_types][j])
			else:
				print('createTrainingTestingDataset() : ERROR')
				sys.exit()
			if objectClassesData[object_types][j] not in complete_list:
				complete_list.append(objectClassesData[object_types][j])

		positives_division = math.ceil(len(positives) / 2.0)
		negatives_division = math.ceil(len(negatives) / 2.0)

		print('Checking object types  : ' + object_types  + ' with objects ' + str(len(positives)))
		if len(positives) == 0:
			continue

		shuffle(positives)
		shuffle(negatives)
		for k in range(len(positives)):
			image_no, presence = getObjectElementInfo(positives[k])
			if image_no in files_removed:
					print('ERROR with files')
					sys.exit()

			if k  < positives_division:
				training_list.append(positives[k])

				

				if image_no not in global_train:
					global_train.append(image_no)
			else:	
				testing_list.append(positives[k])
				if image_no not in global_val:
					global_val.append(image_no)

		for k in range(len(negatives)):
			image_no, presence = getObjectElementInfo(negatives[k])	
			if image_no in files_removed:
					print('ERROR with files')
					sys.exit()


			if k < negatives_division:
				training_list.append(negatives[k])
				if image_no not in global_train:
					global_train.append(image_no)
			else:
				testing_list.append(negatives[k])
				if image_no not in global_val:
					global_val.append(image_no)

		shuffle(training_list)
		shuffle(testing_list)		

		
		training_file   = open(directory + object_types + '_train.txt', 'w')
		test_file       = open(directory + object_types + '_val.txt', 'w')
		train_test_file = open(directory + object_types + '_trainval.txt', 'w')

		str1 = '\n'.join(training_list)
		str2 = '\n'.join(testing_list)
		str3 = '\n'.join(complete_list)

		training_file.write(str1)
		test_file.write(str2)
		train_test_file.write(str3)

		training_file.close()
		test_file.close()
		train_test_file.close()

	final_global_train, final_global_test = checkGlobalTestTrain(global_train, global_val)
	
	global_train_file   = open(directory + 'train.txt', 'w')
	global_test_file    = open(directory + 'val.txt', 'w')

	str1 = '\n'.join(final_global_train)
	str2 = '\n'.join(final_global_test)

	global_train_file.write(str1)
	global_test_file.write(str2)

	global_train_file.close()
	global_test_file.close()
		


def checkGlobalTestTrain(global_train, global_test):
	
	print('image object map length = ' + str(len(image_object_map.keys())))
	print(image_object_map[1410])

	final_global_train = copy.copy(global_train)
	final_global_test = copy.copy(global_test)
	
	list_to_check = final_object_types
	print(list_to_check)
	for i in range(len(global_train)):

		image_no = int(global_train[i])
		if image_no in files_removed:
			print('Removing')
			final_global_train.remove(global_train[i])
			continue

		for j in range(len(image_object_map[image_no])):
			if image_object_map[image_no][j] in list_to_check:
				list_to_check.remove(image_object_map[image_no][j])

	if len(list_to_check) != 0:
		print('ERROR with X')
		sys.exit()

	list_to_check = final_object_types

	for i in range(len(global_test)):

		image_no = int(global_test[i])
		if image_no in files_removed :
			final_global_test.remove(global_test[i])
			continue

		for j in range(len(image_object_map[image_no])):
			if image_object_map[image_no][j] in list_to_check:
				list_to_check.remove(image_object_map[image_no][j])

	if len(list_to_check) != 0:
		print('ERROR with Y')
		sys.exit()
	return final_global_train, final_global_test
		
total_dropped = 0

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
	object_types = []
	image_object_map[i] = []
	new_image_with_bounding_box = convertToGrayScale(im, 640, 480)
	for ix, obj in enumerate(objs):

		name_obj = obj.find('name')
		name = name_obj.text
		bbox = obj.find('bndbox')
		
		for visit_element in bbox.iter('bndbox'):
	
			x1 = int(math.floor(float(bbox.find('xmin').text)))
			x2 = int(math.ceil(float(bbox.find('xmax').text)))
			y1 = int(math.floor(float(bbox.find('ymin').text)))
			y2 = int(math.ceil(float(bbox.find('ymax').text)))

			'''if x1 == 0 or y1 == 0:
				print('ERORR with boxes ' + str(i))
				sys.exit()'''	

			if x1 < 10 or x2 > 630 or y1 < 0 or y2 > 480:
				root.remove(obj)
			elif (x2 - x1) <= 30 and (y2 - y1) <= 30 :
				root.remove(obj)
			elif x1 == 0 and x2 == 0 and y1 == 0 and y2 == 0:
				root.remove(obj)
			elif x2 < x1 or y2 < y1:
				print('ERORR with boxes ' + str(i))
				sys.exit()
			
			else:
				if name not in image_object_map[i]:
					image_object_map[i].append(name)
				if name not in final_object_types:
					final_object_types.append(name)

				name_obj.text = class_dic[name]
				object_types.append(class_dic[name])
				cv2.rectangle(new_image_with_bounding_box, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)
			
		if class_dic[name] not in final_classes:
			final_classes.append(class_dic[name])

	for object_type in object_classes:
		if object_type in object_types:
			str_to_add = str(i) + ' 1'
			if str_to_add not in objectClassesData[object_type]:
				objectClassesData[object_type].append(str_to_add)		
		else:
			str_to_add = str(i) + ' -1'
			if str_to_add not in objectClassesData[object_type]:
				objectClassesData[object_type].append(str_to_add)

	
	if len(object_types) == 0:
		print('Dropping image ' + str(i))
		total_dropped += 1
		files_removed.append(i)
		
	else:
		#save new bounding box image
		cv2.imwrite(save_dir + 'JPEGImages/' + str(i) +'.jpg', im)
		cv2.imwrite(save_dir + 'JPEGImages/' + str(i) +'_bounding.jpg', new_image_with_bounding_box)

		#save xml file
		tree.write(save_dir + 'Annotations/' + str(i) + '.xml')
		
for k in range(total_images):
	if k not in image_object_map.keys():
		print('ERROR with image object map')	
		sys.exit()


saveImageSetData(object_classes, objectClassesData)
print('Total images dropped = ' + str(total_dropped))
print(final_classes)
print('Total classes = ' + str(len(final_classes)))
