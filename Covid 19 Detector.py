#!/usr/bin/env python
# coding: utf-8

# ### 1. Importing Libraries

# In[5]:


import pandas as pd
import os
import shutil


# ### 2. Creating Data for Positive Samples

# In[2]:


FILE_PATH = "metadata.csv"
IMAGES_PATH = 'images'


# In[3]:


df = pd.read_csv(FILE_PATH)
print(df.shape)


# In[4]:


df.head(n=5)


# In[7]:


TARGET_DIR = "Dataset/Covid"

if not os.path.exists(TARGET_DIR):
    os.makedir(TARGET_DIR)
    print("Covid Folder Created Successfully")
else:
    print("Covid Folder Already Exists !")


# In[33]:


cnt = 0

for i,row in df.iterrows():
    if row['finding'] == "COVID-19" and row['view'] == 'PA':
        filename = row['filename']
        image_path = os.path.join(IMAGES_PATH,filename)
        image_copy_path = os.path.join(TARGET_DIR,filename)
        shutil.copy2(image_path,image_copy_path)
        #print("Copying image: ",str(cnt+1))
        cnt+=1
        
print("Total Images Copied: ", cnt)


# In[35]:


# Sampling Images from Kaggle Dataset
import random
KAGGLE_FILE_PATH = "NORMAL"
TARGET_NORMAL_DIR = "Dataset/Normal"


# In[39]:


image_names = os.listdir(KAGGLE_FILE_PATH)


# In[40]:


random.shuffle(image_names)


# In[43]:


cnt = 0
for i in range(142):
    
    image_name = image_names[i]
    image_path = os.path.join(KAGGLE_FILE_PATH,image_name)
    
    target_path = os.path.join(TARGET_NORMAL_DIR,image_name)
    
    shutil.copy2(image_path,target_path)
    #print("Copying image: ",str(cnt+1))
    cnt+=1
    
print("Total Images Copied: ", cnt)


# In[ ]:




