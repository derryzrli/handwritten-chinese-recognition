#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import shutil
import numpy as np
import pandas as pd
import PIL.Image
from matplotlib import pyplot as plt


# ### Look for installed font to show Chinese characters in Matplotlib

# In[2]:


from matplotlib.font_manager import FontProperties
plt.rcParams['font.family'] = ['Heiti TC'] 
plt.rcParams['axes.unicode_minus'] = False  # in case minus sign is shown as box


# In[3]:


import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout  
from tensorflow.keras.optimizers import Adam


# In[4]:


raw_data_path = '/Users/derryzrli/Downloads/data_dsi_capstone/traditional_chinese_characters_cleaned'
training_data_path = '/Users/derryzrli/Downloads/data_dsi_capstone/traditional_chinese_training_data_100/'
testing_data_path = '/Users/derryzrli/Downloads/data_dsi_capstone/traditional_chinese_testing_data_100/'

os.chdir(raw_data_path)
print( 'Current working directory:', os.getcwd() ) 


# ### 100 Most Common Characters in Chinese

# In[11]:


selected_chars = ['的', '二', '是', '不', '了', '在', '人', '有', '我', '他', '這', '個', '們', '中', '來', '上', '大', '為', '和', 
                  '國', '地', '道', '以', '說', '時', '要', '就', '出', '會', '可', '也', '你', '對', '生', '能', '而', '子', '那', '得',
                 '於', '著', '下', '自', '之', '年', '過', '發', '後', '作', '里', '用', '到', '行', '所', '然', '家', '種', '事', '成', 
                '方', '多', '經', '去', '法', '學', '如', '都', '同', '現', '當', '沒', '動', '面', '起', '看', '定', '天', '分', '還', 
                  '進', '好', '小', '部', '其', '些', '主', '樣', '理', '心', '她', '本', '前', '開', '但', '因', '只', '從', '想', '實', '日'] 


# In[6]:


len(set(selected_chars))


# In[12]:


os.chdir(raw_data_path)
try: 
    os.mkdir(training_data_path) 

except:
    shutil.rmtree(training_data_path)
    os.mkdir(training_data_path) 

finally: 
    for char in selected_chars:
        shutil.copytree(raw_data_path+'/'+char, training_data_path+'/'+char )


# ----

# ### Data Augmentation

# #### Training Set

# In[17]:


train_data_gen = ImageDataGenerator(rescale = 1./255, 
                                    validation_split = 0.2,
                                    width_shift_range = 0.05,   
                                    height_shift_range = 0.05,
                                    zoom_range = 0.1)


# In[18]:


train_gen = train_data_gen.flow_from_directory(training_data_path,
                                               target_size = (50,50),
                                               batch_size = 8,
                                               class_mode = 'categorical',
                                               subset = 'training'
                                              )


# #### Testing Set

# In[19]:


test_data_gen = ImageDataGenerator(rescale = 1./255, 
                                    validation_split = 0.2)


# In[20]:


test_gen = test_data_gen.flow_from_directory(training_data_path,
                                             target_size = (50,50),
                                             batch_size = 8,
                                             class_mode = 'categorical',
                                             subset = 'validation')


# In[21]:


test_gen[0][0].shape


# In[30]:


model = Sequential() 
model.add(
    Conv2D(filters=5, 
           kernel_size=(2,2), 
           activation='relu', 
           padding='same',
           input_shape=(50,50,3),
          )
)

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(
     Conv2D(filters=5, 
            kernel_size=(2,2), 
            activation='relu', 
            padding='same',
           )
)

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dropout(rate=0.5))

model.add(Dense(512, activation='relu'))

model.add(Dense(len(selected_chars), activation='softmax'))


# In[31]:


model.summary()


# In[32]:


model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


# In[ ]:


history = model.fit(train_gen, validation_data=test_gen, epochs=1000, verbose=1)


# In[21]:


os.chdir(training_data_path)
model.save('CNN_model_100.h5')


# ### Save the history info to track accuracy and loss

# In[22]:


import pickle


# In[23]:


with open('./trainHistoryDict', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)


# In[24]:


os.getcwd()


# In[ ]:


my_dict = pickle.load(open('./trainHistoryDict', 'rb'))
my_dict

