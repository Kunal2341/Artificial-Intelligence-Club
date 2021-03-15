
# coding: utf-8

# # CAFETERIAI   ⚙️ A Smart Cafe ⚙️
# AI CLUB 
# 
# ALPHARETTA HIGH SCHOOL
# 
# 🍏🍓🍇🥚🥞🍗🍔🍟🍕🌮🍣🍦🍰🥛🍩

# # PART -1 --> ALL INPUTES NEEDED

# In[2]:


#Number of sec that the food video is cut 
#1000 is 1 sec and 2000 is 2 sec
numsec = 1000
#face confidence is oppisite 
#0 is good and 100 is bad
faceconfmin = 0
faceconfmax = 100
#resnet cutting image min prob percentage 
minimum_percentage_probability = 10
#when detecting food if frequency is lower than x it gets deleted
frequency_delete = 3
#when deleteing food the probability limit it 
probability_limit = 75


# # PART 0
# Setup files and libraries

# In[3]:


#!pip3 install dfply


# In[4]:


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
import matplotlib.pyplot as plt
import os
import pandas as pd
import datetime
import random
import glob
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil.parser import parse
import csv
from dfply import *
from IPython.display import Markdown, display
from datetime import date
import pickle


# In[5]:


#!pip install psutil


# # 🔜 🔜 🔜 📹
# 🧐🤔🧐🤔🧐🤔🧐🤔

# # ARE YOU HERE

# # 
# SETUP WEBCAM SOURCE 0 1 2?
# SETUP MAX WAIT TIME

# # 
# SET LOCATION

# In[6]:


#Folder to save video *new
os.chdir('/Users/NidhiAneja/Documents/AI/Cafeteria/OpenCV-Python-Series-master/src_final/final_new_vids_in/')
base_dir ='/Users/NidhiAneja/Documents/AI/Cafeteria/OpenCV-Python-Series-master/src_final/final_new_vids_in/'


# # PART A
# #CAPTURE VIDEO
# BREAK INTO FRAME
# #LOAD FILES IN A COMMON FOLDER UNTIL STUDENT IS IDENTIFIED

# # A1. This function extracts images from video with 1 fps

# In[7]:


def extract_image_one_fps(video_source_path):
    vidcap = cv2.VideoCapture(video_source_path)
    count = 0
    success = True
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*numsec)) # 2 second***   
        success,image = vidcap.read()
        ## Stop when last frame is identified
        image_last = cv2.imread("frame{}.png".format(count-1))
        if np.array_equal(image, image_last):
            break
        cv2.imwrite("frame%d.png" % count, image)     # save frame as PNG file
        print( '{}.sec reading a new frame:{}'.format(count,success))
        count += 1


# In[8]:


#FUNCTION FOR ROTATION
#--------
def rotate_image(image,deg):
    if deg ==90:
        return np.rot90(image)
    if deg ==180:
        return np.rot90(image,2)
    if deg == 270:
        return np.rot90(image,-1) #Reverse 90 deg rotation
#--------  


# In[9]:


def printmd(string):
    display(Markdown(string))
#printmd("<span style='color:blue'>Red text</span>")
def printmd(string):
    display(Markdown(string))
#printmd('**bold**')


# In[10]:


get_ipython().system('pwd')


# # A2. Name & Folders

# In[11]:


student_name = 'CAFETERIA' ## FIXED NAME AT THIS STAGE input()   


# #  🛑 ✋ 🛑 ✋ 🛑 ✋ 🛑 ✋ 🛑 ✋ 🤮
# STOP HERE

# # A3.2 PRE-RECORDED VIDEOS CONVERTION
# (Option B)

# In[12]:


# AUTO POPULATE FILE TYPE (mp4, avi, MOV etc.)
x = os.listdir(os.path.join(base_dir, 'CAFETERIA'))
matching = [s for s in x if "CAFETERIA" in s]
filename, V_FORMAT = os.path.splitext(matching[0])
print("V_FORMAT: ",V_FORMAT)
#-----
cap = cv2.VideoCapture(os.path.join(base_dir, student_name, "CAFETERIA" + V_FORMAT ) )
video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
print("video length:", video_length)
#----
os.chdir(os.path.join(os.path.join(base_dir,  student_name) ) )
extract_image_one_fps('CAFETERIA'+ V_FORMAT)
#----
video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
try: 
    video_length > 96 
    print("VIDEO SIZE ", video_length, " FRAMES")
except ValueError:
    print('Oops! That was an invalid recording. Check the webcam setting and try again')
else:
    print('Thank you.')


# # A4. IDENTIFY PERSON
# openCV_faceRECOGNITION_f

# # A4.a Models

# In[13]:


face_cascade = cv2.CascadeClassifier('/Users/NidhiAneja/Documents/AI/Cafeteria/OpenCV-Python-Series-master/src_final/cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()


# # Face Model (latest!)

# In[14]:


recognizer.read('/Users/NidhiAneja/Documents/AI/Cafeteria/OpenCV-Python-Series-master/src_final/recognizers/trainner.yml') # updated 0307
labels = {}
with open('/Users/NidhiAneja/Documents/AI/Cafeteria/OpenCV-Python-Series-master/src_final/pickles/labels.pickle', 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()} 
print(og_labels)


# # A4.b Read Video Recorded (or Loaded) earlier

# In[15]:


os.chdir(os.path.join(os.path.join(base_dir,  student_name) ) )
os.path.join(os.path.join(base_dir,  student_name), student_name + '_face' + V_FORMAT ) 
feedback_vid = os.path.join(os.path.join(base_dir,  student_name), student_name + '_face' + V_FORMAT ) 


# In[16]:


# RUN VIDEO RECORDED EARLIER
cap = cv2.VideoCapture(os.path.join(os.path.join(base_dir,  student_name), student_name + '_face' + V_FORMAT ))


# # A4.c Identify Student

# In[17]:


video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
print("video_length:",  video_length)
get_ipython().system('pwd')


# # 🚩🚩🚩
# ENSURE FOLLOWING SETTING ARE CORRECT!
# 1. minimum CONFIDENCE  2. scaleFactor 3. minNeighbor 4. minSize

# In[18]:


# THIS CODE DOES NOT DO ROTATION
PICK_NAMES = []
frame_cnt = 0
while(frame_cnt < video_length*.99):
    ret, frame = cap.read()
    try: 
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(frame) #, scaleFactor=1.7, minNeighbors=5) #, minSize=(100, 100))
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]        
            # recognizes
            id_, conf = recognizer.predict(roi_gray) # give label id and confidence
            #---- DEFINE PROBABILITY ----#
            if conf >= faceconfmin and conf < faceconfmax:                
                #print(labels[id_])
                #print(conf)
                print(labels[id_], round(conf,0) )
                PICK_NAMES.append( [labels[id_], conf] ) # ****
                #----
                img_item = "ExtractFaceFrame.png"
                cv2.imwrite(img_item, roi_gray)
                #----
                color = (255,0,0) # BGR 0-255
                stroke = 2
                end_cord_x = x+ w
                end_cord_y = y + h
                cv2.rectangle(frame, (x,y), (end_cord_x, end_cord_y), color, stroke)                
        # Display the resulting frame 
        cv2.imshow('frame', frame)
        frame_cnt += 1
        if cv2.waitKey(2) & 0xFF == ord('q'):
            break
    except ValueError:
        print('Oops! Either the 1st or last frame invalid')
    else:
        print(frame_cnt)
#----        # BREAK IF PICK_NAMES IS EMPTY
if len(PICK_NAMES) ==0:
    print("NO FACE SAMPLE COLLECTED! ")    


# In[19]:


if len(PICK_NAMES) ==0:
    print("NO FACE SAMPLE COLLECTED! ")
else:
    print(PICK_NAMES[:10])   


# # A4.d Resolve Student Identity

# In[20]:


df = pd.DataFrame()
for itm in PICK_NAMES:
    teststring =[]
    teststring.append(itm[0])
    teststring.append(itm[1])
    df = df.append([teststring])
    
df.columns = ['name','probability']
#dftest = dftest.reset_index()


# In[21]:


df= df.round(2) #***


# In[22]:


df = df.sort_values(['name', 'probability'], ascending= True)


# In[23]:


df1 = df.groupby('name')['probability'].median().reset_index()
df1 = df1.sort_values(['probability'], ascending= True)
print(df1)


# In[24]:


df2= df.name.value_counts() # FREQUENCY
df2 = pd.DataFrame(df2)
df2.index.name = 'x'
df2.reset_index(inplace=True)
df2.columns = ['name', 'frequency']
df2.head()
#df2=df2.tail(-1)
#df2


# In[25]:


#final_face = pd.merge(df1, df2, on='name')
#if 'pct' in final_face.columns: final_face.drop(columns='pct')
#final_face.drop(columns='pct')
#final_face.head()


# In[25]:


# ABSOLUTE VALUE
if any(df1["name"] == df2["name"].values[0]):
    student_name = df2["name"].values[0]
    print('>> ', df2["name"].values[0], "<< ", "identity confirmed!", 
          "face recognized", df2['frequency'].values[0], "times out of", video_length, "frames" )
else:
    student_name = "Student_not_identified"
    print('Resolve the Identity Conflict among >>', list(df2.name),  "<<")


# In[26]:


# PROPOTIONAL VALUES ****NEW
df2["pct"] = df2["frequency"] / video_length
print(df2["pct"][0], " percent of time faces was recognized for the highest contender")


# In[27]:


studentid_conf = round(100* df2["frequency"].values[0] / sum(df2["frequency"]),0)
print(studentid_conf, "Calculated Confidence Level 100 is the highest level")


# In[28]:


student_name # FINAL NAME TO GO FORWARD


# In[29]:


get_ipython().run_cell_magic('capture', 'outputface', '# ABSOLUTE VALUE\n#printmd("<span style=\'color:blue\'>Red text</span>")\nif any(df1["name"] == df2["name"].values[0]):\n    student_name = df2["name"].values[0]\n    print(\'>> \', df2["name"].values[0], "<< ", "identity confirmed!", \n          "face recognized", df2[\'frequency\'].values[0], "times out of", video_length, "frames" )\nelse:\n    student_name = "Student_not_identified"\n    print(\'Resolve the Identity Conflict among >>\', list(df2.name),  "<<")\n# PROPOTIONAL VALUES ****NEW\nprint("----------------------------->")\ndf2["pct"] = df2["frequency"] / video_length\nprint(df2["pct"][0], " percent of time faces was recognized for the highest contender")\nstudentid_conf = round(100* df2["frequency"].values[0] / sum(df2["frequency"]),0)\nprint("----------------------------->")\nprint(studentid_conf, "Calculated Confidence Level 100 is the highest level")\nprint("----------------------------->")\nprint(student_name) # FINAL NAME TO GO FORWARD')


# # FEEDBACK LOOP

# # CONTINUE CODE

# 🛑 ✋   🛑 ✋    🛑 ✋ 
# Confirm Identity by a Captured Image

# In[30]:


if student_name == 'Student_not_identified':
    #student_name=input(student_name) # FIXED FOR NOW 🏴🏳️🏴
    student_name="SOME_STUDENT"
print(student_name)


# # PART B 
# LaunchPad

# # B0.

# In[31]:


# ONE TIME ONLY
import os
import numpy as np
import cv2
import datetime
import random


# In[32]:


os.chdir('/Users/NidhiAneja/Documents/AI/Cafeteria/OpenCV-Python-Series-master/src_final/')
from utils import CFEVideoConf, image_resize #currently stores in JUSTIN/src folder


# In[33]:


base_dir = '/Users/NidhiAneja/Documents/AI/Cafeteria/OpenCV-Python-Series-master/src_final/final_ph_out/'
video_in = '/Users/NidhiAneja/Documents/AI/Cafeteria/OpenCV-Python-Series-master/src_final/final_ph_out/in/'
image_out = '/Users/NidhiAneja/Documents/AI/Cafeteria/OpenCV-Python-Series-master/src_final/final_ph_out/out/'


# In[34]:


os.chdir(os.path.join(base_dir, 'in') )
os.getcwd()


# # B1. Move Files to Student Folder

# In[35]:


# MOVE AND RENAME THE FOLDER
# ------------------------------
shutil.move('/Users/NidhiAneja/Documents/AI/Cafeteria/OpenCV-Python-Series-master/src_final/final_new_vids_in/CAFETERIA', video_in)


# In[36]:


now = str(datetime.datetime.now())
os.getcwd()


# In[37]:


#matching = [s for s in os.listdir(".") if student_name in s]
nm_check = [s for s in os.listdir(".") if student_name == s]
if nm_check == []:
    os.rename('CAFETERIA', student_name)
elif nm_check[0] == student_name:
    os.rename(student_name, student_name + str(round(random.random(),2)) )
    os.rename('CAFETERIA', student_name)


# In[38]:


video_name = "".join((str(student_name), V_FORMAT))
video_name


# In[39]:


save_path =  os.path.join(base_dir, 'in', student_name, video_name )
save_path


# # PART C
# IMAGEAI OBJECT SEPARATION
# IMAGEAI_OBECTION_DETECTION3_f

# # 
# C0.

# In[40]:


from imageai.Prediction import ImagePrediction 
from imageai.Detection import ObjectDetection
detector = ObjectDetection()


# In[41]:


os.chdir(os.path.join(base_dir, 'in', student_name) )
execution_path = os.path.join(base_dir, 'in', student_name)
execution_path


# # C1. Model Selection

# In[42]:


# MODEL option A
detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath('/Users/NidhiAneja/Documents/AI/IMAGE_AI/ImageAI-master/MODELS/resnet50_coco_best_v2.0.1.h5')
detector.loadModel()


# # C2. Object Separation

# In[43]:


get_ipython().system('pwd')
student_name


# In[44]:


lst_frames = os.listdir()
lst_frames = sorted(lst_frames)
# cleanup 1
if '.DS_Store' in lst_frames: lst_frames.remove('.DS_Store')
lst_frames


# In[45]:


# cleanup 2 #zero bytes file
for filename in os.listdir(os.path.join(video_in, student_name)):
     if os.path.getsize(filename) == 0:
            os.remove(filename) 
            print("File Removed!")


# In[46]:


# FRAMES LIST UPDATED HERE FOR NEXT LOOP
lst_frames=[]
for filename in os.listdir(os.path.join(video_in, student_name)):
    if filename.startswith("frame"):
        lst_frames.append(filename)
lst_frames = sorted(lst_frames)
#lst_frames


# In[47]:


execution_path


# # 🚩🚩🚩 
# CONFIRM THE PROBABILITY TO SPLIT THE PLATE

# # Try Custom Objects! 0311

# In[48]:


custom_objects = detector.CustomObjects(bottle = True, 
                                        cup = True,   #fork = True,   knife = True,   spoon = True,   bowl = True,   
                                        banana = True,   
                                        apple = True,   
                                        sandwich = True,   
                                        orange = True,
                                        broccoli = True,   
                                        carrot = True,     
                                        pizza = True,   
                                        donot = True,   
                                        cake = True,
                                        hot_dog = True,
                                        bowl = True,
                                        book = True)
# hotdog = True,  cellphone  = True


# In[49]:


print(lst_frames)
for lst in lst_frames:
    print(lst)
    try:
        detections = detector.detectCustomObjectsFromImage(custom_objects = custom_objects, 
                                           input_image= os.path.join( execution_path, lst),
                                           output_image_path= lst[5] + str(random.randint(0,100)), 
                                           minimum_percentage_probability= minimum_percentage_probability,
                                           extract_detected_objects=True)
    except:
        pass
    else:
        print("PARSING IMAGES DONE!  ", lst)

len(lst_frames)


# In[50]:


object_folders = [x[0] for x in os.walk(execution_path)][1:] 
object_folders = sorted(object_folders)


# In[51]:


# GIVE RANDOM FILE NAMES BEFORE MOVING TO A SINGLE FOLDER
i = 0
for i in range(len(object_folders)):
    path = object_folders[i]
    #print(os.listdir(path))
    for filename in os.listdir(path):
        os.rename(path  + '/'+ filename, 
                  path  + '/captured'  +   str(random.randint(1,10001))  +'.jpg')
        i = i +1


# In[52]:


#IDENTIFY ZERO BYTES FILES AND REMOVE THOSE  # cleanup 2 #zero bytes file        
REMOVE=0
for j in range(len(object_folders)):
    path = object_folders[j]
    for filename in os.listdir(path):
        if os.path.getsize(os.path.join(path, filename) ) < 20000:
            os.remove(os.path.join(path, filename) )
            REMOVE = +1
print(REMOVE, " Removed!")         


# In[53]:


#IDENTIFY ZERO BYTES FILES AND REMOVE THOSE  # cleanup 2 #zero bytes file        
REMOVE=0
for j in range(len(object_folders)):
    path = object_folders[j]
    for filename in os.listdir(path):
        if os.path.getsize(os.path.join(path, filename) ) > 80000:
            os.remove(os.path.join(path, filename) )
            REMOVE = +1
print(REMOVE, " Removed!")         


# In[54]:


# create objects folder to move all captured images in one folder
if not any(os.listdir(execution_path)) == "objects":
    os.mkdir("objects")


# In[55]:


# MOVE FILES FROM object-n folders to object folder
for i in range(len(object_folders)):
    object_files = os.listdir(object_folders[i])
    if '.DS_Store' in object_files: object_files.remove('.DS_Store')
    for f in object_files:
        try:
            shutil.move(os.path.join(object_folders[i], f), os.path.join(execution_path, "objects"))
        except:
            pass


# # PART D: CUSTOM MODEL PREDICTION

# # D1. Load Custom Model 

# In[56]:


#!pip3 install https://github.com/OlafenwaMoses/ImageAI/releases/download/2.0.2/imageai-2.0.2-py3-none-any.whl
get_ipython().system('pwd')


# In[57]:


from imageai.Prediction.Custom import ModelTraining
from imageai.Prediction.Custom import CustomImagePrediction
prediction = CustomImagePrediction()
prediction.setModelTypeAsResNet()


# # 🚩🚩🚩 
# CONFIRM MODEL FOLDER & MODEL VERSION BELOW!
# CONFIRM NUMBER OF FOOD ITEMS!! 

# In[58]:


custom_model_path='/Users/NidhiAneja/Documents/AI/IMAGE_AI/ImageAI-master/custom/'
os.chdir(os.path.join(custom_model_path, "models"))
os.listdir()


# In[59]:


number_of_classes = 14


# In[60]:


#*** 14 ITEMS *** 0303
prediction.setModelPath('/Users/NidhiAneja/Documents/AI/IMAGE_AI/ImageAI-master/custom/models/model_ex-042_acc-0.985008.h5') 
prediction.setJsonPath('/Users/NidhiAneja/Documents/AI/IMAGE_AI/ImageAI-master/custom/json/model_class.json')
#-----
prediction.loadModel(num_objects=14) #updated 0303


# # D2. Predict Food Items 

# In[61]:


detected_path = os.path.join(execution_path, "objects")
all_files = os.listdir(detected_path)
if '.DS_Store' in all_files: all_files.remove('.DS_Store')
all_files[:2]


# In[62]:


all_images_array = []
for each_file in all_files:
    if(each_file.endswith(".jpg") or each_file.endswith(".png")):
        all_images_array.append(each_file)
all_images_array[:2]


# In[63]:


os.chdir(detected_path)
#============================
# *** RUN PREDICTIONS ***
#============================
results_array = prediction.predictMultipleImages(all_images_array, result_count_per_image=1)


# # D3. Summarize Results

# In[64]:


save_results = []
for i in results_array:
    FoodItem, prob = i["predictions"], i["percentage_probabilities"]
    for idx in range(len(FoodItem)):
        #print(pred[idx] , " : " , prob[idx])
        save_results.append( (FoodItem[idx], prob[idx]) )
    #print("-----------------------")


# In[65]:


df = pd.DataFrame(save_results, columns=['FoodItem','Probability'])
len(df)


# In[66]:


df["image_name"] = all_images_array
df.Probability = pd.to_numeric(df.Probability, errors = 'coerce').fillna(0).astype(np.int64)


# In[67]:


file_ext = str(random.randint(1,100))
df.to_csv(os.path.join(execution_path, "save_results" + file_ext  + ".csv"), index=False, encoding = 'utf8')


# In[68]:


df = df.sort_values([ 'FoodItem', 'Probability'], ascending= False)
df.head(2)


# In[69]:


dfx = df


# In[70]:


itm_tags = dfx.FoodItem.value_counts()
itm_tags = pd.DataFrame(itm_tags)
itm_tags.index.name = 'x'
itm_tags.reset_index(inplace=True)
itm_tags.columns = ['FoodItem', 'frequency']
itm_tags


# In[71]:


FoodItem_excl_lst = []
for excl_lst in itm_tags[(itm_tags['frequency'] <= frequency_delete)]['FoodItem']:
    FoodItem_excl_lst.append(excl_lst)


# In[72]:


print("deleting --->")
(FoodItem_excl_lst)


# # Probability Threshold

# In[73]:


print(len(df))
df.Probability = pd.to_numeric(df.Probability, errors = 'coerce')
df = df.sort_values(['Probability'], ascending = False)
df = df.drop(df[df.Probability < probability_limit].index)
print(len(df))
df


# In[74]:


df = df.drop_duplicates('FoodItem')
df


# In[75]:


df = df[~df['FoodItem'].isin(FoodItem_excl_lst)]
df


# In[76]:


now = str(datetime.datetime.now())
#---
df['dates'] = (now[:10])
df['timestamp'] = (now[11:19])
df['student_name'] = student_name
df


# # D4. Price Table

# In[109]:


# ONE TIME  ## ONLY 15 ITEMS ***
# REF- food folders:: https://drive.google.com/open?id=1zSOhOZWVygKn08tctel-Xfwnjq_nlaYE 
ItemCostTable = {"FoodItem" : [
    "Cheerios",
    "CheezIts","Chips","CinamonToastCrunch",
    "FruitSnacks", "GoldenGrahamsBar", "GoldenGrahamsCereal", 
    "McNuggets", "MilkBlue","MilkPurple", "NutriGrain",
    "Pizza", "QuarterPounder", "RiceKrispes"],
                 "Cost": [1,2,3,4,5,6,7,8,9,10,11,12,13,14]}
CostTable = pd.DataFrame(ItemCostTable)          
CostTable.head(4)


# In[78]:


if not number_of_classes  == len(CostTable):
    print("! UPDATE PRICE TABLE !")
    


# In[79]:


df = pd.merge(df, CostTable, on=['FoodItem', 'FoodItem'])


# In[80]:


df


# # Add column

# In[81]:


df['studentid_conf'] = studentid_conf


# In[82]:


df['file_extn'] = file_ext


# # D5. Save CSV File **FINAL**

# In[83]:


df.to_csv(os.path.join(execution_path, "final_safe_results" + file_ext  + ".csv"), 
          index=False, encoding = 'utf8')


# # D6. VISUALS

# In[84]:


os.path.join(video_in, student_name, "objects", df["image_name"].iloc[0])


# # 
# **SHOW ONLY FINAL ITEMS DETECTED***

# In[85]:


len(df)


# In[86]:


for i in range(len(df)):
    im = cv2.imread(os.path.join(video_in, student_name, "objects", 
                                 df["image_name"].iloc[i]))
    im_resized = cv2.resize(im, (224, 224), interpolation=cv2.INTER_LINEAR)
    plt.imshow(cv2.cvtColor(im_resized, cv2.COLOR_BGR2RGB))
    print(df["image_name"].iloc[i])
    print(df["FoodItem"].iloc[i])
    print(df["Probability"].iloc[i])
    plt.show()


# # 
# **SHOW ALL ITEMS***

# In[87]:


itm_tags = dfx.FoodItem.value_counts() # FREQUENCY
itm_tags = pd.DataFrame(itm_tags)
itm_tags.index.name = 'x'
itm_tags.reset_index(inplace=True)
itm_tags.columns = ['FoodItem', 'frequency']
itm_tags


# In[88]:


# show everything
for i in range(len(dfx)):
    im = cv2.imread(os.path.join(video_in, student_name, "objects", 
                                 dfx["image_name"].iloc[i]))
    im_resized = cv2.resize(im, (224, 224), interpolation=cv2.INTER_LINEAR)
    plt.imshow(cv2.cvtColor(im_resized, cv2.COLOR_BGR2RGB))
    print(dfx["image_name"].iloc[i])
    print(dfx["FoodItem"].iloc[i])
    print(dfx["Probability"].iloc[i])
    plt.show()


# In[89]:


os.path.join(video_in, student_name, "objects")


# In[90]:


os.chdir(os.path.join(video_in, student_name, "objects"))
capt_lst = os.listdir(".")
if '.DS_Store' in capt_lst: capt_lst.remove('.DS_Store')
len(capt_lst)
for capt in capt_lst:
    for d2 in dfx['FoodItem'][dfx['image_name'] == capt]:
        os.rename(capt, d2+str(random.randint(1,10000)) + '.jpg')


# # PART Z
# APPEND FOLDER w/ a random Number so same student can be re-entered 

# In[91]:


'''
with open('/Users/NidhiAneja/Documents/AI/Cafeteria/OpenCV-Python-Series-master/src_final/final_ph_out/CafeRecords.csv','w') as csvfile:
    fieldnames = ['FoodItem', 'Probability', 'image_name', 'dates', 'timestamp', 'student_name', 'Cost', 'studentid_conf', 'file_extn']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
'''


# In[92]:


with open('/Users/NidhiAneja/Documents/AI/Cafeteria/OpenCV-Python-Series-master/src_final/final_ph_out/CafeRecords.csv','a') as f:
    f.write('\n')
    df.to_csv(f, header=False, index = False, encoding='utf-8')


# In[93]:


# ENDING WITH ADDING DATE TO THE FOLDER
now[:10]
file_ext = str(random.randint(1,1000))
os.rename(os.path.join(video_in, student_name ), 
          os.path.join(video_in, student_name + "_" + file_ext ) )  #str(now[:10]) ) )
print(file_ext)


# #  🛑 ✋    🛑 ✋    🛑 ✋

# # Part R

# In[94]:


CafeRecords = pd.read_csv('/Users/NidhiAneja/Documents/AI/Cafeteria/OpenCV-Python-Series-master/src_final/final_ph_out/CafeRecords.csv', error_bad_lines=False)
CafeRecords['dates'] = pd.to_datetime(CafeRecords['dates']) 
CafeRecords.tail(4)


# In[95]:


#Current_Spend = CafeRecords >> mask(X.student_name == student_name, X.dates == datetime.today().strftime('%Y-%m-%d')) >> group_by(X.student_name) >> summarize(Total_Spend = X.Cost.sum() )
Current_Spend = CafeRecords >> mask(X.student_name == student_name, X.dates == date.today().strftime('%Y-%m-%d')) >> group_by(X.student_name) >> summarize(Total_Spend = X.Cost.sum() )
if Current_Spend.empty:
    Current_Spend  = pd.DataFrame(columns = ["student_name", "Total_Spend"])
    Current_Spend=  {'student_name' : student_name, "Total_Spend" : 0}


# In[96]:


Current_Spend


# In[97]:


Semester_Spend = CafeRecords >> mask(X.student_name == student_name, X.dates > '2019-01-01') >> group_by(X.student_name) >> summarize(Total_Spend = X.Cost.sum() )


# In[98]:


purchase_hist = CafeRecords >> mask(X.student_name == student_name) >> group_by(X.student_name, X.FoodItem) >> summarize(Semester_Count = n(X.FoodItem) )
purchase_hist = list(zip(purchase_hist.FoodItem, purchase_hist.Semester_Count))


# In[99]:


print('YOUR ACTIVITY SUMMARY WITH CAFE AI...')
print("------------->")
print('STUDENT NAME == ', student_name)
print("--------------->")
print('FACE RECOGINITION MODEL CONFIDENCE == %.lf' %(df['studentid_conf'][0]))
print("----------------->")
print("YOU BOUGHT (Prob)", list(zip(df.FoodItem,df.Probability)))
print("------------------>")
#print('DOLLARS SPEND $', Current_Spend[1])
print('TOTAL CHARGES TODAY == $%.lf' %(Current_Spend['Total_Spend'] ) )
print("------------------->")
#print('CURRENT SEMESTER SPEND', Semester_Spend[1])
print('TOTAL DEBIT THIS SEMESTER == $%.lf' %(Semester_Spend['Total_Spend'] ) )
print("--------------------->")
print('YOUR PURCHASE IN THIS SEMSTER ',  purchase_hist)
print("----------------------->")

