import datetime
import sys
import matplotlib.pyplot as plt

optimizer = Adam(0.0002, 0.5)
#  We use a pre-trained VGG19 model to extract image features from the high resolution and the generated high resolution images and minimize the mse between them
vgg = VGG()
vgg.trainable = False
vgg.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

# Calculate output shape of D (PatchGAN)
patch = int(height / 2**4)
disc_patch = (patch, patch, 1)

# Build and compile the discriminator
D = Discriminator()
D.summary()
D.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

# Build the generator
G = Generator()
G.summary()
# High res. and low res. images
img_hr = Input(shape=shape)
img_lr = Input(shape=shape)

# Generate high res. version from low res.
fake_hr = G(img_lr)

# Extract image features of the generated img
fake_features = vgg(fake_hr)

# For the combined model we will only train the generator
D.trainable = False

# Discriminator determines validity of generated high res. images
validity = D(fake_hr)

combined = Model([img_lr, img_hr], [validity, fake_features])
combined.compile(loss=['binary_crossentropy', 'mse'], loss_weights=[1e-3, 1], optimizer=optimizer)
