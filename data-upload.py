import cv2
import os
import numpy as np

files_name = os.listdir('Data_x/Train/Noisy_images')
train_x = []   #hazy images
train_y = []   #sharp images

for _,i in enumerate(files_name):
  if(os.path.isfile(os.path.join('Data_x/Train/Noisy_images',i))):
    train_x = np.append(train_x,cv2.imread(os.path.join('Data_x/Train/Noisy_images',i))/255.0)
    train_y = np.append(train_y, cv2.imread(os.path.join('Data_x/Train/Ground_Truths',i))/255.0)
    
    
train_x = np.asarray(train_x).reshape(-1, 256,256, 3)
train_y = np.asarray(train_y).reshape(-1, 256,256, 3)
print('train_x', train_x.shape)
print('train_y', train_y.shape)
train_x=train_x.astype(np.float32)
train_y=train_y.astype(np.float32)
