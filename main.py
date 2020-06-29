import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
import sys

test_images = []

resolution = '512 by 512 Resolution'
results_directory='SR_Final/' + resolution 


    
def plot_images(input_images, generated_images, ori_img):
  plt.figure(1, figsize=(13,13))
  for i in range(5):
    plt.subplot(1, 5, i+1).set_title('Noisy Image # {}' .format(i+1))
    plt.imshow(input_images[i, :, :, :])
  plt.savefig(save_path+'/Noisy-Image.png')
  plt.figure(2, figsize=(13,13))
  
  for i in range(5):
    plt.subplot(1, 5, i+1).set_title('Generated Image # {}' .format(i+1))
    plt.imshow(generated_images[i, :, :, :])
  plt.savefig(save_path+'/Generated-Image.png')
  #plt.figure(3, figsize=(13,13))
  
  #for i in range(5):
   # plt.subplot(1, 5, i+1).set_title('Original Image # {}' .format(i+1))
   # plt.imshow(ori_img[i, :, :, :])
   
  plt.show()
  plt.close()


epoches = 50
batch_size = 2
valid = np.ones((batch_size,) + disc_patch)
fake = np.zeros((batch_size,) + disc_patch)
batches = int(train_x.shape[0]/batch_size)
for epoch_number in range(epoches):
  print('Epoch : {}/{}'.format(epoch_number+ 1, epoches))
  for batch_number in range(batches):
    imagenumber=batch_number+1
    
    Ground_Truths = train_y[batch_number*batch_size:(batch_number+1)*batch_size]
    Hazy_Images = train_x[batch_number*batch_size:(batch_number+1)*batch_size]
    fake_hr = G.predict(Noisy_Images)
    #   Train the discriminators (original images = real / generated = Fake)
    d_loss_real = D.train_on_batch(Ground_Truths, valid)
    d_loss_fake = D.train_on_batch(fake_hr, fake)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
#   ------------------
#   Train Generator
#   ------------------

#   Sample images and their conditioning counterparts
#   The generators want the discriminators to label the generated images as real
#   Extract ground truth image features using pre-trained VGG19 model
    image_features = vgg.predict(Ground_Truths)
  
#   Train the generators
    g_loss = combined.train_on_batch([Noisy_Images, Ground_Truths], [valid, image_features])
    sys.stdout.write('\r[Batch {0}/{1}], [D-loss: {2}, D-acc: {3}], [G-loss: {4}]'.format(batch_number+1, batches, d_loss[0], d_loss[1], np.mean(g_loss)))
  #   Rescale images 0 - 1
  print('\n')
  if epoch_number % 1 == 0:
    save_path = os.path.join(results_directory, str(epoch_number+1))
    if not os.path.exists(save_path):
      os.makedirs(save_path)
    G.save(save_path + '/Generator')
    test = test_x[:]
    gen_img = G.predict(test)
    gen_img = 0.5*gen_img+0.5
    ori_img = test_y[:]
    plot_images(test, gen_img, ori_img)
    

    
