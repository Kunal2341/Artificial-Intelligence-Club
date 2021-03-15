
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import time
import os
import shutil
from PIL import Image
import psutil
import random
import datetime
from pprint import pprint
#--------
import cv2
import os
import numpy as np
import pickle
import datetime
import glob
import matplotlib.pyplot as plt
import shutil


# In[2]:


random.seed()
now = str(datetime.datetime.now())
now = now[:10] +"_"+ str(round(random.random(),3) )
#---
base_dir ='/Users/NidhiAneja/Documents/AI/Cafeteria/OpenCV-Python-Series-master/src_final/Scripts/'
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
os.chdir(os.path.join('/Users/NidhiAneja/Documents/AI/Cafeteria/OpenCV-Python-Series-master/src_final/Scripts/', student_name) )
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
recording_length = 5
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
#extract_image_one_fps(os.path.join(base_dir,  student_name, mp4_file ) )
#-----


# In[3]:


#Folder to save video *new
os.chdir(os.path.join('/Users/NidhiAneja/Documents/AI/Cafeteria/OpenCV-Python-Series-master/src_final/Scripts/', student_name))
base_dir =os.path.join('/Users/NidhiAneja/Documents/AI/Cafeteria/OpenCV-Python-Series-master/src_final/Scripts/', student_name)


# In[4]:



def extract_image_one_fps(video_source_path):
    vidcap = cv2.VideoCapture(video_source_path)
    count = 0
    success = True
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*1000)) # 2 second***   
        success,image = vidcap.read()
        ## Stop when last frame is identified
        image_last = cv2.imread("frame{}.png".format(count-1))
        if np.array_equal(image, image_last):
            break
        cv2.imwrite("frame%d.png" % count, image)     # save frame as PNG file
        print( '{}.sec reading a new frame:{}'.format(count,success))
        count += 1


# In[5]:


#function to detect face using OpenCV
def detect_face(img):
    #convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #load OpenCV face detector, I am using LBP which is fast
    #there is also a more accurate but slow Haar classifier
    face_cascade = cv2.CascadeClassifier('/Users/NidhiAneja/Documents/AI/Cafeteria/OpenCV-Python-Series-master/src_final/lbpcascades/lbpcascade_frontalface.xml')
    #let's detect multiscale (some images may be closer to camera than others) images
    #result is a list of faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    #if no faces are detected then return original img
    if (len(faces) == 0):
        return None, None
    #under the assumption that there will be only one face,
    #extract the face area
    (x, y, w, h) = faces[0]
    #return only the face part of the image
    return gray[y:y+w, x:x+h], faces[0]


# In[6]:


# VIDEO TO FACE EXTRACTION


# In[7]:


video_path = (base_dir)
videos = os.listdir(base_dir)
if '.DS_Store' in videos: videos.remove('.DS_Store')        
video_path


# In[8]:


i = 0
for i in range(len(videos)):
    os.listdir(os.path.join(video_path) )

    os.chdir(os.path.join(video_path))

    '''
    X = os.listdir(".")
    if '.DS_Store' in X: X.remove('.DS_Store')
    X[0]
    '''
    extract_image_one_fps(os.path.join(video_path, videos[i]) ) 

    # cleanup 2 #zero bytes file
    for filename in os.listdir(os.path.join(video_path) ):
         if os.path.getsize(filename) == 0:
                os.remove(filename) 

    os.chdir(os.path.join(video_path))
    photo_path = os.path.join(video_path)
    face_frames = os.listdir(os.path.join(video_path))
    if '.DS_Store' in face_frames: face_frames.remove('.DS_Store')


    # Video file removed
    # for face_frames1 in face_frames:#os.listdir(os.path.join(video_in, student_name)):
    #    if face_frames1.endswith("mp4") or  face_frames1.endswith("avi"):
    #        os.remove(face_frames1) 

    face_frames = os.listdir(os.path.join(video_path))
    if '.DS_Store' in face_frames: face_frames.remove('.DS_Store')        

    k=0
    for FRAMES in face_frames:
        tmp = cv2.imread(os.path.join(photo_path, FRAMES) )
        #tmp = rotate_image(tmp, 270) # DONT ROTATE
        face, rect = detect_face(tmp)
        k += 1
        img_nm =  videos[i] + str(k) + ".png"
        cv2.imwrite(img_nm, face)

    # Frames removed
    for filename in os.listdir(os.path.join(video_path) ):
        if filename.startswith("frame"):
            os.remove(filename) 

    # cleanup 2 #zero bytes file
    for filename in os.listdir(os.path.join(video_path) ):
         if os.path.getsize(filename) == 0:
                os.remove(filename) 
                print("File Removed!")        

