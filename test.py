import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
from keras.models import load_model

resolution = '512 by 512 Resolution'
results_dir='/SR_Final/' + resolution 
#epochs = [40, 160, 360, 480]
epochs = [1, 10, 20, 25]

Indoor_directory='/SR_Final/Data/NTIRE 2018/Indoor Images'
Test_path = os.path.join(Indoor_directory, 'Test', 'Hazy Images') 
width = height = 256

test_images = []

#for img_name in os.listdir(Test_path):
#  img = cv2.imread(os.path.join(Test_path, img_name))
#  img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
#  img = img/127.5 - 1
#  img = img.astype(np.float32)
#  test_images.append(img)
  
test_images = test_x 


Results = []
Results.append(0.5 * test_images + 0.5)
for epoch in epochs:
  print('Epoch: {}'.format(epoch))
  path = os.path.join(results_dir, str(epoch+1))
  G = load_model(os.path.join(path,'Generator.h5'))

  gen_imgs = G.predict(test_images)
  Results.append(0.5 * gen_imgs + 0.5)
  
  
Results = np.array(Results).reshape(-1, width, height, 3)

labels = ['100', '200', '300', '400']
# Plot Generated Images 
rows, cols = 5, 5
fig, axs = plt.subplots(rows, cols, figsize=(10, 10))
img_count = 0
for row in range(rows):
  for col in range(cols):
    axs[row, col].imshow(Results[img_count, :, :, :])
    axs[row, col].axis('off')
    if row == 0:
      axs[row, col].set_title('Noisy image# {}'.format(col+1))
    else:
      axs[row, col].set_title('Gen-Epoch# {}'.format(labels[row-1]))
    img_count += 1
    
# fig.subplots_adjust(top = 0.99, bottom=0.01, hspace=0.01, wspace=0.2)
fig.savefig('/SR_Final/Results_256') 
plt.show()
plt.close()
    
