from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.layers import LeakyReLU
from keras.applications import VGG19
channels = 3
height = 256   
width = 256   
shape = (height, width, channels)

residual_blocks = 16

def Residual_Block(inputs, filters):
  """Residual block described in paper"""
  conv = Conv2D(filters, kernel_size=3, strides=1, padding='same')(inputs)
  conv = Activation('relu')(conv)
  conv = BatchNormalization(momentum=0.8)(conv)
  conv = Conv2D(filters, kernel_size=3, strides=1, padding='same')(conv)
  conv = BatchNormalization(momentum=0.8)(conv)
  conv = Add()([conv, inputs])
  return conv

def Generator():
#   Low resolution image input
  inputs = Input(shape=shape)
  
#   Pre-residual block
  conv1 = Conv2D(64, kernel_size=9, strides=1, padding='same')(inputs)
  conv1 = Activation('relu')(conv1)

#   Propogate through residual blocks
  res_output = Residual_Block(conv1, 64)
  for _ in range(residual_blocks - 1):
    res_output = Residual_Block(res_output, 64)
  
#   Post-residual block
  conv2 = Conv2D(64, kernel_size=3, strides=1, padding='same')(res_output)
  conv2 = BatchNormalization(momentum=0.8)(conv2)
  conv2 = Add()([conv2, conv1])
  
  conv3 = Conv2D(256, kernel_size=3, strides=1, padding='same')(conv2)
  conv3 = Activation('relu')(conv3)
  
  conv4 = Conv2D(256, kernel_size=3, strides=1, padding='same')(conv3)
  conv4 = Activation('relu')(conv4)
  
#   Generate high resolution output
  outputs = Conv2D(channels, kernel_size=9, strides=1, padding='same', activation='tanh')(conv4)
  return Model(inputs = inputs, outputs = outputs)

def Discriminator():
  inputs = Input(shape)
  conv1 = Conv2D(64, (4, 4), strides=1, padding='same')(inputs)   
  conv1 = LeakyReLU(alpha=0.2)(conv1)

  conv2 = Conv2D(64, (4, 4), strides=2, padding='same')(conv1)   # 64
  conv2 = LeakyReLU(alpha=0.2)(conv2)
  conv2 = BatchNormalization()(conv2)
  
  conv3 = Conv2D(128, (4, 4), strides=1, padding='same')(conv2)   # 64
  conv3 = LeakyReLU(alpha=0.2)(conv3) 
  conv3 = BatchNormalization()(conv3)
  
  conv4 = Conv2D(128, (4, 4), strides=2, padding='same')(conv3)   # 32
  conv4 = LeakyReLU(alpha=0.2)(conv4) 
  conv4 = BatchNormalization()(conv4)

  conv5 = Conv2D(256, (4, 4), strides=1, padding='same')(conv4)   # 32
  conv5 = LeakyReLU(alpha=0.2)(conv5)
  conv5 = BatchNormalization()(conv5)
  
  conv6 = Conv2D(256, (4, 4), strides=2, padding='same')(conv5)   # 16
  conv6 = LeakyReLU(alpha=0.2)(conv6)
  conv6 = BatchNormalization()(conv6)
  
  conv7 = Conv2D(512, (4, 4), strides=1, padding='same')(conv6)   # 16
  conv7 = LeakyReLU(alpha=0.2)(conv7)
  conv7 = BatchNormalization()(conv7)
  
  conv8 = Conv2D(512, (4, 4), strides=2, padding='same')(conv7)   # 8
  conv8 = LeakyReLU(alpha=0.2)(conv8)
  conv8 = BatchNormalization()(conv8)
  
  outputs = Dense(1024)(conv8)
  outputs = LeakyReLU(alpha=0.2)(outputs)
  outputs = Dense(1, activation='sigmoid')(outputs)

  return Model(inputs=inputs, outputs=outputs)

def VGG():
#   Builds a pre-trained VGG19 model that outputs image features extracted at the third block of the model
  vgg = VGG19(weights="imagenet")
  
#   Set outputs to outputs of last conv. layer in block 3
#   See architecture at: https://github.com/keras-team/keras/blob/master/keras/applications/vgg19.py
  vgg.outputs = [vgg.layers[9].output]
  inputs = Input(shape=shape)
  
#   Extract image features
  outputs = vgg(inputs)
  return Model(inputs=inputs, outputs=outputs)
      
