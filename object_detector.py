import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse


class ObjectDetector:

    CLASSES = ()

    NETS = ()

    

    def __init__(self, gpu_id=0, cpu_mode=False, demo_net='zf'):


	self.CLASSES =  ('__background__', 'health', 'monster', 'high_grade_weapons', 'high_grade_ammo', 'other_ammo', 'my_shoots','monster_shoots')
	self.NETS    =  {'vgg16': ('VGG16','VGG16_faster_rcnn_final.caffemodel'),
               'zf': ('ZF','zf_faster_rcnn_iter_100000.caffemodel')}

	cfg.TEST.HAS_RPN = True  

    	prototxt = os.path.join(cfg.MODELS_DIR, self.NETS[demo_net][0],'faster_rcnn_end2end', 'test.prototxt')
    	caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models', self.NETS[demo_net][1])

    	if not os.path.isfile(caffemodel):
        	raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    	if cpu_mode:
       		caffe.set_mode_cpu()
    	else:
        	caffe.set_mode_gpu()
        	caffe.set_device(gpu_id)
       		cfg.GPU_ID = gpu_id
    	self.net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    	print '\n\nLoaded object detection network {:s}'.format(caffemodel)


    def getBoundingBoxInfo(self, im):
	
	    # Detect all object self.CLASSES and regress object bounds
	    timer = Timer()
	    timer.tic()
	    scores, boxes = im_detect(self.net, im)

	    timer.toc()
	    
	    # Visualize detections for each class
	    CONF_THRESH = 0.8
	    NMS_THRESH = 0.3

	    count = 0
	    bounding_boxes_per_image = {}

	    for cls_ind, cls in enumerate(self.CLASSES[1:]):

		cls_ind += 1 # because we skipped background
		cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
		cls_scores = scores[:, cls_ind]
		dets = np.hstack((cls_boxes,
		                  cls_scores[:, np.newaxis])).astype(np.float32)
		keep = nms(dets, NMS_THRESH)
		dets = dets[keep, :]

		##### check the detections for each class
		inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
	    	if len(inds) == 0:
			continue

	    	for i in inds:
			bbox = dets[i, :4]
			score = dets[i, -1]
			bounding_boxes_per_image[count] = [bbox[0], bbox[1], bbox[2] , bbox[3] ]

			count += 1
	    	
	    return bounding_boxes_per_image

