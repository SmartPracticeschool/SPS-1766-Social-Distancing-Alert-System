# -*- coding: utf-8 -*-
"""
@author: Jal
"""
# base path to YOLO directory
MODEL_PATH = "yolo-coco"

# initialize minimum probability to filter weak detections along with
# the threshhold when appling non-maxima suppression
MIN_CONF = 0.3
NMS_THRESH = 0.3

# boolean indicating if NVIDIA CUDA GPU should be used
USE_GPU = False

# define the maximum safe distance (in pixels) that two people can be
# from each other
MIN_DISTANCE = 50