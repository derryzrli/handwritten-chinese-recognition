import os
import shutil
import numpy as np
import pandas as pd
import PIL.Image
from matplotlib import pyplot as plt
import pickle

from matplotlib.font_manager import FontProperties
plt.rcParams['font.family'] = ['Heiti TC'] 
plt.rcParams['axes.unicode_minus'] = False  # in case minus sign is shown as box

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout  
from tensorflow.keras.optimizers import Adam

raw_data_path = '/Users/derryzrli/Downloads/data_dsi_capstone/traditional_chinese_characters_cleaned'
training_data_path = '/Users/derryzrli/Downloads/data_dsi_capstone/traditional_chinese_training_data/'
testing_data_path = '/Users/derryzrli/Downloads/data_dsi_capstone/traditional_chinese_testing_data/'

os.chdir(raw_data_path)
print( 'Current working directory:', os.getcwd() ) 

complete_chars = os.listdir(os.getcwd())
complete_chars.remove('.DS_Store')

train_data_gen = ImageDataGenerator(rescale = 1./255, 
                                    validation_split = 0.2,
                                    width_shift_range = 0.05,   
                                    height_shift_range = 0.05,
                                    zoom_range = 0.1)

train_gen = train_data_gen.flow_from_directory(training_data_path,
                                               target_size = (50,50),
                                               batch_size = 8,
                                               class_mode = 'categorical',
                                               subset = 'training')

test_data_gen = ImageDataGenerator(rescale = 1./255, 
                                    validation_split = 0.2)

test_gen = test_data_gen.flow_from_directory(training_data_path,
                                             target_size = (50,50),
                                             batch_size = 8,
                                             class_mode = 'categorical',
                                             subset = 'validation')

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
model.add(Dense(len(complete_chars), activation='softmax'))

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

history = model.fit(train_gen, validation_data=test_gen, epochs=50)
print(history.history)

with open('./trainHistoryDictComplete', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

os.chdir(training_data_path)
model.save('CNN_model_complete.h5')
