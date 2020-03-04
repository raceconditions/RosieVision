######## Webcam Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 10/2/19
# Description: 
# This program uses a TensorFlow Lite model to perform object detection on a
# video. It draws boxes and scores around the objects of interest in each frame
# from the video.
#
# This code is based off the TensorFlow Lite image classification example at:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py
#
# I added my own method of drawing boxes and labels using OpenCV.

# Import packages
import os
import cv2
import numpy as np
import sys
import importlib.util

import traceback
import logging
from tflite_runtime.interpreter import Interpreter
from tflite_runtime.interpreter import load_delegate

logging.getLogger().setLevel(logging.INFO)

class ObjectDetector(object):
    
    def __init__(self, model_path, label_path, use_coral_flag, use_tpu_flag, res_x, res_y, min_conf_threshold):

        self.res_y = res_y
        self.res_x = res_x
        self.use_coral_flag = use_coral_flag
        if use_coral_flag:
            from edgetpu.detection.engine import DetectionEngine
            from edgetpu.utils import dataset_utils
        self.min_conf_threshold = min_conf_threshold

        # Load the label map
        with open(label_path, 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]
        
        if self.labels[0] == '???':
            del(self.labels[0])
        
        if use_tpu_flag:
            self.interpreter = Interpreter(model_path=model_path, experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
        else:
            self.interpreter = Interpreter(model_path=model_path)
        
        self.interpreter.allocate_tensors()
        
        # Get model details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.height = self.input_details[0]['shape'][1]
        self.width = self.input_details[0]['shape'][2]
        
        self.is_floating_model = (self.input_details[0]['dtype'] == np.float32)
        
        self.input_mean = 127.5
        self.input_std = 127.5
        
        #Coral
        if use_coral_flag:
            self.engine = DetectionEngine(model_path)
            self.labels = dataset_utils.read_label_file(label_path)
            _, height, width, _ = self.engine.get_input_tensor_shape()

    def apply_coral_model(self, input_data):
        print("here")
        ans = self.engine.detect_with_input_tensor(input_data, threshold=0.05, top_k=10)
        print("here2")
        for obj in ans:
            if self.labels:
                print(self.labels[obj.label_id])
            print('score = ', obj.score)
            box = obj.bounding_box.flatten().tolist()
            print('box = ', box)
    
    def apply_tflite_model(self, input_data):
        # Perform the actual detection by running the model with the image as input
        self.interpreter.set_tensor(self.input_details[0]['index'],input_data)
        self.interpreter.invoke()
    
        # Retrieve detection results
        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0] # Bounding box coordinates of detected objects
        classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0] # Class index of detected objects
        scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0] # Confidence of detected objects
    
        return (boxes, classes, scores)
    
    def process_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (self.width, self.height))
        input_data = np.expand_dims(frame_resized, axis=0)
    
        # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
        if self.is_floating_model:
            input_data = (np.float32(input_data) - self.input_mean) / self.input_std
    
        if self.use_coral_flag:
            self.apply_coral_model(input_data)
            scores = []
        else:
            (boxes, classes, scores) = self.apply_tflite_model(input_data)

        return (frame, boxes, classes, scores)

    def is_interesting_object(self, scores, classes):
        is_interesting_object = False
        interesting_classes = []
        for i in range(len(scores)):
            if ((scores[i] > self.min_conf_threshold) and (scores[i] <= 1.0)):
                is_interesting_object = True
                interesting_classes.append(self.labels[int(classes[i])])
        return is_interesting_object, interesting_classes


    def draw_frame(self, frame, boxes, classes, scores):
        # Loop over all detections and draw detection box if confidence is above minimum threshold
        for i in range(len(scores)):
            if ((scores[i] > self.min_conf_threshold) and (scores[i] <= 1.0)):
    
                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                ymin = int(max(1,(boxes[i][0] * self.res_y)))
                xmin = int(max(1,(boxes[i][1] * self.res_x)))
                ymax = int(min(self.res_y,(boxes[i][2] * self.res_y)))
                xmax = int(min(self.res_x,(boxes[i][3] * self.res_x)))
                
                cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 4)
                
                # Draw label
                object_name = self.labels[int(classes[i])] # Look up object name from "labels" array using class index
                label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
        (flag, encodedImage) = cv2.imencode(".jpg", frame)
        return encodedImage
