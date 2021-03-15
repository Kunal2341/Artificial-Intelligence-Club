#!/usr/bin/env python
# coding: utf-8

#============================================================
# CAPTURE STILL IMAGES USING python script
# 02/27/19
#============================================================

#!python capture_multiplePHOTOS_f.py --name='CPP' --webcam='0'

#=========================
# import the necessary packages
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--name",   required=True, help="names of the student")
ap.add_argument("-w", "--webcam", required=True, help="webcam source 0 1 2")
args = vars(ap.parse_args()) 
# display a friendly message to the user
print("Hi there {}, it's nice to meet you!".format(args["name"]))
print("webcam source {}".format(args["webcam"]))
#=========================
import cv2
import os
import random
# set folder to save pictures.
os.chdir('/Users/NidhiAneja/Documents/AI/Cafeteria/OpenCV-Python-Series-master/src_final/Scripts')
base_dir = os.getcwd()
base_dir
#----
student_name = args["name"]
#----
if not os.path.exists(student_name):
        os.makedirs(student_name)
os.chdir(os.path.join(base_dir, student_name))
os.getcwd()
#----
cam = cv2.VideoCapture(int(args["webcam"]) )
cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)
#----
cv2.namedWindow("CAFETERAI")
cv2.resizeWindow("CAFETERAI", 2, 1)
cv2.moveWindow("CAFETERAI", 10,10)
img_counter = 1000 + random.randint(1,10000)
while True:
    ret, frame = cam.read()
    cv2.imshow("CAFETERAI", frame)
    if not ret:
        break
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        #img_name = "opencv_frame_{}.png".format(img_counter)
        img_name = "{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
	
	#print("test")
        print("{} written!".format(img_name))
        img_counter += 1
        #print("FFFFF")
#=========================        
cam.release()
cv2.destroyAllWindows()
#=========================
