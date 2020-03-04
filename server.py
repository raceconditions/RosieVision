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

import traceback
import logging
import time
from multiprocessing import Pool, Queue, Manager

from object_detector import ObjectDetector
from motion_detector import MotionDetector
#from web_server import WebServer
from web_server_flask import WebServer
import argparse

logging.getLogger().setLevel(logging.INFO)

parser = argparse.ArgumentParser(description='Process mpjpg video stream for motion and object detection.')
parser.add_argument("video_path", help="URL of the video stream to process", default="http://io:8081/stream.mjpg")
parser.add_argument("--web-port", help="Port to run the web server on", type=int, default=8081)
parser.add_argument("--confidence", help="Minimal confidence percent for object detection", type=float, default=0.6)
parser.add_argument("--use-tpu", help="Leverage coral TPU", action='store_true')
args = parser.parse_args()

WEB_PORT=args.web_port
MODEL_NAME = "coco"
#GRAPH_NAME = "detect.tflite"
LABELMAP_NAME = "labelmap.txt"
VIDEO_PATH = args.video_path
MIN_CONF_THRESHOLD = args.confidence

USE_CORAL = False
USE_TPU = args.use_tpu
if USE_TPU:
    GRAPH_NAME = 'detect_edgetpu.tflite'
else:
    GRAPH_NAME = 'detect.tflite'


CWD_PATH = os.getcwd()
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

logging.info("Web Port:   %s", WEB_PORT)
logging.info("Model:      %s", PATH_TO_CKPT)
logging.info("Labels:     %s", PATH_TO_LABELS)
logging.info("Video Path: %s", VIDEO_PATH)
logging.info("Use TPU:    %s", USE_TPU)
logging.info("Use Coral:  %s", USE_CORAL)
logging.info("Confidence: %s", MIN_CONF_THRESHOLD)

def motion_detector_started(pid):
    logging.info("Started motion detector in process %s", pid)

def start_video():
    # Open video file
    video = cv2.VideoCapture(VIDEO_PATH)
    imW = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    imH = video.get(cv2.CAP_PROP_FRAME_HEIGHT)

    logging.info("Detected video with dimensions %s x %s", imW, imH)
    if imW == 0.0 or imH == 0.0:
        logging.info("Frame detected as empty, trying again later...")
        return

    logging.info("Starting video...")
    while(video.isOpened()):
        try:
            ret, frame = video.read()
            (frame, boxes, classes, scores) = object_detector.process_frame(frame)
            (is_interesting_object, interesting_classes) = object_detector.is_interesting_object(scores, classes)
            motion_detector.process_frame(frame, is_interesting_object, interesting_classes)
            mjpg_frame = object_detector.draw_frame(frame, boxes, classes, scores)
            web_server.frame_available(mjpg_frame)
            #new_frame_queue.put(frame.copy())
        except Exception:
            logging.exception("Unknown failure processing frame.")

    # Clean up
    video.release()
    #pool.close()
    #pool.join()

if __name__ == '__main__':
    object_detector = ObjectDetector(PATH_TO_CKPT, PATH_TO_LABELS, USE_CORAL, USE_TPU, 1296, 730, MIN_CONF_THRESHOLD)
    web_server = WebServer(WEB_PORT)

    #manager = Manager()
    #new_frame_queue = manager.Queue()
    motion_detector = MotionDetector(0, 1296, 0, 730, 1296, 730)
    web_server.start()

    while(True):
        try:
            start_video()
        except Exception:
            logging.exception("Failure processing video.")
        time.sleep(120)

    #pool = Pool(1)
    #pool.apply_async(motion_detector.start_processing, args=(new_frame_queue, ), callback=motion_detector_started)

