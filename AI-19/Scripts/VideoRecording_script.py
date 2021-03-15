#!/usr/bin/env python
# coding: utf-8
#=========================================================
# VIDEO CAPTURE + CONVERT TO FRAMES + SAVING TO A FOLDER 
#=========================================================
# SETABLES
# Number of frames
# number of seconds of video
# video format
# folder for saving videos
# video source 0 / 1 / 2 
# frame extraction frequency
# 
#=========================================================
# DEFAULT VALUES


import cv2
import time
import numpy as np
import datetime
import random
import os
random.seed()
now = str(datetime.datetime.now())
now = now[:10] +"_"+ str(round(random.random(),3) )
#---
base_dir ='/Users/NidhiAneja/Documents/AI/Cafeteria/OpenCV-Python-Series-master/src_final/'
os.chdir(base_dir)
#---
def extract_image_one_fps(video_source_path):
    vidcap = cv2.VideoCapture(video_source_path)
    count = 0
    success = True
    while success:
      vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*1000)) # 2 second***   
      success,image = vidcap.read()
      image_last = cv2.imread("frame{}.png".format(count-1))
      if np.array_equal(image, image_last):
          break
      cv2.imwrite("frame%d.png" % count, image)     # save frame as PNG file
      print( '{}.sec reading a new frame:{}'.format(count,success))
      count += 1
#------ # record and convert to still images
#---#student_name = input()
student_name = 'CAFETERIA'+"__" + now
#---
if not os.path.exists(student_name):
        os.makedirs(student_name)
#---
nm1 = str(student_name)
nm3 = '.mp4'
mp4_file = "".join((nm1, nm3))
#----
os.chdir(os.path.join('/Users/NidhiAneja/Documents/AI/Cafeteria/OpenCV-Python-Series-master/src_final/demo/', student_name) )
#----
frames_per_second = 10    # 24.0 # HIGHER NUMBER LOWER # OF FRAMES!
#----
#my_res = '480p'
my_res = '720p'
#----
STD_DIMENSIONS = {
    "480p": (640, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "4k": (3840, 2160),
}
# Video encoding, see www.fourcc.org/codecs.php for more codecs
VIDEO_TYPE = {
    'avi': cv2.VideoWriter_fourcc(*'XVID'),
    'mp4': cv2.VideoWriter_fourcc(*'XVID'),
}
#---
def get_video_type(filename):
    filename, ext = os.path.splitext(filename)
    if ext in VIDEO_TYPE:
        return VIDEO_TYPE[ext]
    return VIDEO_TYPE['avi']
#---
#def set_res(cap, res='480p'):
def set_res(cap, res='720p'):    
    #width, height = STD_DIMENSIONS['480p']
    width, height = STD_DIMENSIONS['720p']
    if res in STD_DIMENSIONS:
        width, height = STD_DIMENSIONS[res]
    cap.set(3, width)
    cap.set(4, height)
    return width, height
#---
recording_length = 15
cap = cv2.VideoCapture(0)
t0 = time.time()
#---
dims = set_res(cap, my_res)
video_type_cv2 = get_video_type(mp4_file)
#---
mp4_out = cv2.VideoWriter(mp4_file, video_type_cv2, frames_per_second, dims)
#---
while True:
    ret, frame = cap.read()
    mp4_out.write(frame)
    cv2.imshow('frame', frame)
    if time.time() > (t0 + recording_length) or  cv2.waitKey(20) & 0xFF == ord('q'):
        break
# When everything done, release the capture
cap.release()
mp4_out.release()
cv2.destroyAllWindows()
#-----
extract_image_one_fps(os.path.join(base_dir,  student_name, mp4_file ) )
#-----

