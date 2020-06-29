# SRGAN-Low-dose-CT-Scan-Denoising
Low-dose CT Scan Image denoising/super-resolution through deep learning-based Generative adversarial networks (GANs) specifically SRGAN. This project uses a deep learning-based SRGAN model to remove noise from low-dose CT Scans and upscale 16x16 CT Scans by a 4x factor. The resulting 64x64 images display sharp features that are plausible based on the dataset that was used to train the neural net.
Here's a random, example of what this network can do. From left to right, the first column is the ground-truth input image with very low noise (Normal Dose Image), the second one is a low dose image with much more Gaussian and Poisson noise (Heavily downsampled), the third is the output generated by the neural net. As you can see, the network is able to produce a very plausible reconstruction of the ground truth images.
![Normal](https://user-images.githubusercontent.com/37848312/85992849-e905c780-ba0e-11ea-812c-0354ac9bbc1a.jpg)
![Noisy](https://user-images.githubusercontent.com/37848312/85992293-18680480-ba0e-11ea-9ed6-c3d72a4adaa7.png)
![SRGAN](https://user-images.githubusercontent.com/37848312/85992326-228a0300-ba0e-11ea-9985-e1b7af4c0031.png)





# Requirements

You will need Python 3 with Tensorflow, numpy, scipy and i implemented of GoogleColab so you dont have to worry about anything.

## Dataset

After you have the required software above you will also need the Lung nodule dataset`https://luna16.grand-challenge.org/data/`. 
To validate the performance of the model I used a real clinical dataset of Lung Image Database Consortium (LIDC). The dataset contains 5,335 normal dose CT
scan images of size 512x512. LDCT image dataset isn’t available. To make dataset low dose i randomly induced Poison Noise and Gaussian Noise of different distribution. As most of the LDCT images are effected by Phantom poison noise and Gaussian noise. After making the dataset low dose we have HR images and their counterpart LR images. There are around 3,335 images for training the model and 2000 images for validation in our dataset. The testing images distinct from training images. 

# Training the model

Training with default settings: `python3 main.py --run train`. The script will periodically output an example batch in PNG format onto the `/train` folder, and checkpoint data will be stored in the `/checkpoint` folder.



# About the author
Saqib Nazir
[LinkedIn profile of Saqib Nazir](https://www.linkedin.com/in/saqib-nazir-149467186/).
