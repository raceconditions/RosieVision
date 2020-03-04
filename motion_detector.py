# CarSpeed Version 2.0

# import the necessary packages
from pathlib import Path
from pytz import timezone
import time
import math
import datetime
import cv2
import pytz
import logging 
import sys
import os
import requests
import threading
import json

from multiprocessing import Queue

logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
rootLogger = logging.getLogger()
rootLogger.setLevel(logging.DEBUG)
logPath = "/media/images"
fileName = "motion"
media_dir = "/media/images/"

fileHandler = logging.FileHandler("{0}/{1}.log".format(logPath, fileName))
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)

# the following enumerated values are used to make the program more readable
WAITING = 0
TRACKING = 1
BASING = 2
UNKNOWN = 0
LEFT_TO_RIGHT = 1
RIGHT_TO_LEFT = 2
ANALYSIS_SCALE_FACTOR = 2

# state maintains the state of the speed computation process
# if starts as WAITING
# the first motion detected sets it to TRACKING
 
class MotionDetector(object):

    # define timezones
    EASTERN = timezone('US/Eastern')
    UTC = pytz.utc
     
    SAVE_CSV = True
    THRESHOLD = 64#15
    MIN_AREA = 1000
    BLURSIZE = (15,15)
    SHOW_BOUNDS = True
    
    # hard coded monitor area
    MONITOR_TOP_LEFT_X = 300
    MONITOR_TOP_LEFT_Y = 600
    MONITOR_BOT_RIGHT_X = 1340
    MONITOR_BOT_RIGHT_Y = 900
    

    def __init__(self, left_x, right_x, upper_y, lower_y, resolution_x, resolution_y):
        # if it is tracking and no motion is found or the x value moves
        # out of bounds, state is set to SAVING and the speed of the object
        # is calculated
        # self.initial_x holds the x value when motion was first detected
        # self.last_x holds the last x value before tracking was was halted
        # depending upon the direction of travel, the front of the
        # vehicle is either at x, or at x+w 
        # (tracking_end_time - tracking_start_time) is the elapsed time
        # from these the speed is calculated and displayed 
        
        rootLogger.info("Initalizing motion detector")
        self.state = WAITING
        self.initial_x = 0
        self.last_x = 0
        self.last_base_readjust = None
        self.last_motion = None
        self.last_message = None
        self.base_image = None
        self.secs = 0.0
        self.current_fps = 6
        self.is_recording_video = False
        self.is_interesting_object = False
        self.interesting_classes = []
        self.video_out = None
        self.video_frame_count = 0
        
        if self.SAVE_CSV:
            self.csvfileout = media_dir + "motion.csv"
            if not Path(media_dir + "motion.csv").is_file():
                self.record_motion('Date,Day,Time,Image,Classes')
        else:
            self.csvfileout = ''
        
        self.upper_left_x = left_x
        self.lower_right_x = right_x
        self.upper_left_y = upper_y
        self.lower_right_y = lower_y

        self.resolution_x = resolution_x
        self.resolution_y = resolution_y
             
        self.monitored_width = self.lower_right_x - self.upper_left_x
        self.monitored_height = self.lower_right_y - self.upper_left_y
        self.monitored_area = self.monitored_width * self.monitored_height
        self.MAX_AREA = self.monitored_area * 0.3

        rootLogger.debug("Monitored area:")
        rootLogger.debug(" upper_left_x {}".format(self.upper_left_x))
        rootLogger.debug(" upper_left_y {}".format(self.upper_left_y))
        rootLogger.debug(" lower_right_x {}".format(self.lower_right_x))
        rootLogger.debug(" lower_right_y {}".format(self.lower_right_y))
        rootLogger.debug(" monitored_width {}".format(self.monitored_width))
        rootLogger.debug(" monitored_height {}".format(self.monitored_height))
        rootLogger.debug(" monitored_area {}".format(self.monitored_area))
        rootLogger.debug("Initialized values:")
        rootLogger.debug(" state {}".format(self.state))

        self.last_timestamp = datetime.datetime.now().timestamp()
        self.iteration_diffs = []

    def start_processing(self, inbound_frame_queue):
        self.inbound_frame_queue = inbound_frame_queue

        processing_thread = threading.Thread(target = self.watch_queue)
        processing_thread.start()
        return os.getpid()
        
    def watch_queue(self):
        self.is_running = True

        while self.is_running:
            print(self.inbound_frame_queue.qsize())
            inbound_frame = self.inbound_frame_queue.get()
            self.process_frame(inbound_frame)

    def stop(self):
        self.is_running = False

    # calculate elapsed seconds
    def secs_diff(self, endTime, begTime):
        diff = (endTime - begTime).total_seconds()
        return diff

    # record speed in .csv format
    def record_motion(self, res):
        f = open(self.csvfileout, 'a')
        f.write(res+"\n")
        f.close

    def send_rosie_request(self, interesting_classes):
        try:
            payload = {'objects': interesting_classes}
            requests.post(url="http://rosie:8081/motion_detected",data=json.dumps(payload))
        except:
            print("send request failed")

    def initialize_video(self):
        rootLogger.info("Initializing video")
        currenttimestamp = datetime.datetime.now(self.UTC).astimezone(self.EASTERN)
        dateFolder = media_dir + currenttimestamp.strftime("%Y%m%d") + "/"
        imageFilename = "motion_at_" + currenttimestamp.strftime("%Y%m%d_%H%M%S%Z") + ".avi"

        if not os.path.exists(dateFolder):
            os.makedirs(dateFolder)
        if self.SAVE_CSV:
            cap_time = currenttimestamp
            self.record_motion(cap_time.strftime("%Y.%m.%d")+','+cap_time.strftime('%A')+','+\
               cap_time.strftime('%H%M')+','+imageFilename+','+"|".join(self.interesting_classes))
        self.video_out = cv2.VideoWriter(dateFolder + '/' + imageFilename,cv2.VideoWriter_fourcc(*'DIVX'), self.current_fps, (self.resolution_x, self.resolution_y))

    def add_image_timestamp(self, image):
        currenttimestamp = datetime.datetime.now(self.UTC).astimezone(self.EASTERN)
        cv2.rectangle(image, (0, self.resolution_y - 35), (720, self.resolution_y), (0,0,0), -1)
        cv2.putText(image, "{0:.1f}fps - {1}".format(self.get_current_fps(), currenttimestamp.strftime("%A %d %B %Y %I:%M:%S%p %Z")),
            (10, image.shape[0] - 10), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 1)

    def save_image(self, image):
        self.video_frame_count = self.video_frame_count + 1
        self.video_out.write(image)

    def close_video(self):
        self.is_interesting_object = False

        if self.video_frame_count > 0:
            rootLogger.info("Closing video, saving %s frames", self.video_frame_count)
        self.video_frame_count = 0
        try:
            if self.video_out:
                self.video_out.release()
        except:
            rootLogger.exception("Unable to release video")

    def get_current_fps(self):
        return self.current_fps

    def process_frame(self, bgr_frame, is_interesting_object, interesting_classes):
        self.is_interesting_object = is_interesting_object
        self.interesting_classes = interesting_classes
        #initialize the timestamp
        timestamp = datetime.datetime.now()

        #track frame processing rate
        self.iteration_diffs.append(timestamp.timestamp() - self.last_timestamp)
        self.last_timestamp = timestamp.timestamp()
        if len(self.iteration_diffs) > 10:
            self.iteration_diffs = self.iteration_diffs[-10:]
            self.current_fps = len(self.iteration_diffs) / sum(self.iteration_diffs)
     
        # grab the raw NumPy array representing the image 
        image = bgr_frame
   
#start here
        #resize frame to speed up motion analysis
        gray = cv2.resize(image,(int(self.resolution_x/ANALYSIS_SCALE_FACTOR),int(self.resolution_y/ANALYSIS_SCALE_FACTOR)))
        # crop area defined by [y1:y2,x1:x2]
        #gray = image[self.upper_left_y:self.lower_right_y,self.upper_left_x:self.lower_right_x]

        # convert the fram to grayscale, and blur it
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, self.BLURSIZE, 0)
     
        # if the base image has not been defined, initialize it
        if self.base_image is None:
            self.base_image = gray.copy().astype("float")
            self.last_base_readjust = timestamp
            self.state = BASING
      
        # compute the absolute difference between the current image and
        # base image and then turn eveything lighter gray than THRESHOLD into
        # white
        frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(self.base_image))
        thresh = cv2.threshold(frameDelta, self.THRESHOLD, 255, cv2.THRESH_BINARY)[1]
        #thresh = cv2.adaptiveThreshold(frameDelta,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

        # dilate the thresholded image to fill in any holes, then find contours
        # on thresholded image
        thresh = cv2.dilate(thresh, None, iterations=2)
        (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
        # look for motion 
        motion_found = False
        biggest_area = 0

        #add timestamp
        self.add_image_timestamp(image)
     
        # examine the contours, looking for the largest one
        if self.state != BASING:
            for c in cnts:
                (x1, y1, w1, h1) = cv2.boundingRect(c)
                # get an approximate area of the contour
                found_area = w1*h1 
                # find the largest bounding rectangle
                if (self.MIN_AREA < found_area < self.MAX_AREA) and (found_area > biggest_area):  
                    biggest_area = found_area
                    motion_found = True
                    x = x1
                    y = y1
                    h = h1
                    w = w1
                    #draw rectangle for every contour found greater than MIN_AREA
                    adjusted_x = (self.upper_left_x + x) * ANALYSIS_SCALE_FACTOR #account for border line drawing
                    adjusted_y = (self.upper_left_y + y) * ANALYSIS_SCALE_FACTOR
                    #cv2.drawContours(image, [c], 0, (0, 255, 0), 2), offset=(self.upper_left_x, self.upper_left_y)) 
                    cv2.rectangle(image, (adjusted_x, adjusted_y), (adjusted_x + w, adjusted_y + h),(0,255,0),2)

        if motion_found:
            self.last_motion = timestamp
            if self.is_interesting_object and not self.is_recording_video:
                self.initialize_video()
                try:
                    #notify rosie that we've detected movement
                    if self.last_message == None or self.secs_diff(timestamp, self.last_message) > 30:
                        rosie_thread = threading.Thread(target = self.send_rosie_request, args = (self.interesting_classes))
                        rosie_thread.start()
                        self.last_message = timestamp
                except Exception as e:
                    print("thread exception", e)

            if self.state == WAITING:
                # intialize tracking
                self.state = TRACKING
                self.initial_x = x
                self.last_x = x
                self.initial_time = timestamp
                rootLogger.info("Start tracking")
            else:
                # compute the lapsed time
                self.secs = self.secs_diff(timestamp,self.initial_time)
    
                if self.secs >= 15:
                    self.state = WAITING
                    motion_found = False
                    biggest_area = 0
                    self.base_image = None
                    rootLogger.debug('Resetting')
                    return self.state            
    
            if self.is_recording_video:       
                try:
                    save_thread = threading.Thread(target = self.save_image, args=[image])
                    save_thread.start()
                except Exception as e:
                    print("thread exception:", e)
                    ##just ignore this for now
        else:
            if self.state != WAITING:
                rootLogger.info("tracking stopped %s - %s", self.state, WAITING)
                self.state = WAITING

            #wait for 3 seconds of no motion before closing video
            if self.last_motion is not None and self.secs_diff(timestamp, self.last_motion) > 3:            
                self.close_video()
                
        # only update image and wait for a keypress when waiting for motion
        # This is required since waitkey slows processing.
        if (self.state == WAITING):    
                
            # Adjust the self.base_image as lighting changes through the day
            if self.secs_diff(timestamp, self.last_base_readjust) > 300:
                rootLogger.info("adjusting base image")
                self.last_x = 0
                #cv2.accumulateWeighted(gray, self.base_image, 0.25)
                self.base_image = None
        return self.state
