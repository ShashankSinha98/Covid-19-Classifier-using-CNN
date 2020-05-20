#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


get_ipython().system('wget "http://cb.lk/covid_19"')


# In[2]:


get_ipython().system('unzip covid_19')


# In[ ]:


TRAIN_PATH = "CovidDataset/Train"
VAL_PATH = "CovidDataset/Val"


# In[4]:


import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.layers import *
from keras.models import *
from keras.preprocessing import image


# In[ ]:


## CNN Based Model in Keras

model = Sequential()
model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(224,224,3)))
model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128,kernel_size=(3,3),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[6]:


model.summary()


# In[ ]:


# Training 
train_datagen =  image.ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True
)

test_datagen = image.ImageDataGenerator(rescale=1./255)


# In[8]:


train_generator = train_datagen.flow_from_directory(
    'CovidDataset/Train',
    target_size = (224,224),
    batch_size = 32,
    class_mode = 'binary'
)


# In[9]:


train_generator.class_indices


# In[10]:


validation_generator = test_datagen.flow_from_directory(
    'CovidDataset/Val',
    target_size = (224,224),
    batch_size = 32,
    class_mode = 'binary'
)


# In[11]:


hist = model.fit_generator(
    train_generator,
    steps_per_epoch = 8,
    epochs = 10,
    validation_data = validation_generator,
    validation_steps = 2
)


# In[ ]:


# TODO-
# Class Activation Maps
# Grad CAM


# In[12]:


# Visualization
import matplotlib.pyplot as plt

h = hist.history
plt.style.use('seaborn')
plt.plot(h['loss'],label='Training Loss')
plt.plot(h['val_loss'],label='Validation Loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()


# In[13]:


plt.style.use('seaborn')
plt.plot(h['accuracy'],label='Training Accuracy')
plt.plot(h['val_accuracy'],label='Validation Accuracy')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()


# In[ ]:


## Plot confusion Matrix
from pathlib import Path
dirs = Path(TRAIN_PATH).glob("*")
train_imgs = []
label = []
for each_dir in dirs:

  files = each_dir.glob("*")
  for each_file in files:
    img = image.load_img(each_file,target_size=(224,224))
    img_arr = image.img_to_array(img)/255.0
    train_imgs.append(img_arr)
    if str(each_dir).split("/")[-1] == "Normal":
      label.append(1)
    else:
      label.append(0)


# In[ ]:


train_imgs = np.array(train_imgs)


# In[71]:


print(train_imgs.shape)


# In[ ]:


from sklearn.utils import shuffle
train_imgs,label = shuffle(train_imgs,label,random_state=2)


# In[ ]:


outputs = model.predict_classes(train_imgs)


# In[ ]:



### Use Directly - adapted from http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
import itertools
import matplotlib.pyplot as plt
import numpy as np
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# In[75]:


from sklearn.metrics import confusion_matrix
cnf_matrix = confusion_matrix(outputs,label)
print(cnf_matrix)


# In[ ]:





# In[76]:


plot_confusion_matrix(cnf_matrix,classes=["Covid","Normal"],title="Training Confusion Matrix")


# In[ ]:


## Plot confusion Matrix
dirs = Path(VAL_PATH).glob("*")
test_imgs = []
test_label = []
for each_dir in dirs:

  files = each_dir.glob("*")
  for each_file in files:

    img = image.load_img(each_file,target_size=(224,224))
    img_arr = image.img_to_array(img)/255.0
    test_imgs.append(img_arr)
    if str(each_dir).split("/")[-1] == "Normal":
      test_label.append(1)
    else:
      test_label.append(0)


# In[ ]:


test_imgs = np.array(test_imgs)


# In[79]:


print(test_imgs.shape)


# In[ ]:


test_imgs,test_label = shuffle(test_imgs,test_label,random_state=2)
preds = model.predict_classes(test_imgs)


# In[81]:


test_cnf_matrix = confusion_matrix(preds,test_label)
print(test_cnf_matrix)


# In[82]:


plot_confusion_matrix(test_cnf_matrix,classes=["Covid","Normal"],title="Testing Confusion Matrix")


# In[ ]:




