from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")
import seaborn as sns
import matplotlib.pyplot as plt
import ipywidgets



import os
import sys
import random
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import cv2

from tqdm import tqdm_notebook, tnrange
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

#from training import
from IoU_metric import mean_iou

# Setting some of the parameters

im_width = 128
im_height = 128
im_chan = 1

path_train = '../input/train/'
path_test = '../input/test/'

### Exploring the data

ids = ['1f1cc6b3a4','5b7c160d0d','6c40978ddf','7dfdf6eeb8','7e5a6e5013']

ax1 = plt.figure(1, figsize=(20, 10))
for j, img_name in enumerate(ids):
    q = j + 1
    img = load_img('../input/train/images/' + img_name + '.png')
    img_mask = load_img('../input/train/masks/' + img_name + '.png')

    plt.subplot(1, 2*(1+len(ids)), q*2-1)
    plt.imshow(img)
    plt.subplot(1,2*(1+len(ids)), q*2)
    plt.imshow(img_mask)
plt.savefig('../output/showing_images.png')

train_ids = next(os.walk(path_train + "images"))[2]
test_ids = next(os.walk(path_test + "images"))[2]

##Get and resize train images and masks

# X_train = np.zeros((len(train_ids), im_height, im_width, im_chan), dtype=np.uint8)
# Y_train = np.zeros((len(train_ids), im_height, im_width, 1), dtype=np.bool)
# print('Getting and resizing train images and masks')
# sys.stdout.flush()
# for n, id_ in tqdm_notebook(enumerate(train_ids), total=len(train_ids)):
#     path = path_train
#     img = load_img(path + '/images/' + id_)
#     x = img_to_array(img)[:, :, 1]
#     x = resize(x, (128, 128, 1), mode='constant', preserve_range=True)
#     X_train[n] = x
#     mask = img_to_array(load_img(path + '/masks/' + id_))[:, :, 1]
#     Y_train[n] = resize(mask, (128, 128, 1), mode='constant', preserve_range=True)
#
# print('Done!')

### Check if training data looks all right

# ix = random.randint(0, len(train_ids))
# ax2 = plt.figure()
# plt.imshow(np.dstack((X_train[ix],X_train[ix],X_train[ix])))
# plt.savefig('../output/showing_train_data_x.png')
#
# tmp = np.squeeze(Y_train[ix]).astype(np.float32)
# plt.imshow(np.dstack((tmp,tmp,tmp)))
# plt.savefig('../output/showing_train_data_y.png')

### Define Intersection over Union (IoU) metric

# def mean_iou(y_true, y_pred):
#     prec = []
#     for t in np.arange(0.5, 1.0, 0.05):
#         y_pred_ = tf.to_int32(y_pred > t)
#         score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
#         K.get_session().run(tf.local_variables_initializer())
#         with tf.control_dependencies([up_opt]):
#             score = tf.identity(score)
#         prec.append(score)
#     return K.mean(K.stack(prec), axis=0)

### Build U-Net model

# inputs = Input((im_height, im_width, im_chan))
# s = Lambda(lambda x: x/255)(inputs)
#
# c1 = Conv2D(8, (3, 3), activation='relu', padding='same')(s)  ## two consecutive Convolutional layers
# c1 = Conv2D(8, (3, 3), activation='relu', padding='same')(c1)  ## c1: output tensor of Convolutional Layers
# p1 = MaxPooling2D((2, 2))(c1)  ## p1: output tensor of Max Pooling Layers
#
# c2 = Conv2D(16, (3, 3), activation='relu', padding='same')(p1)
# c2 = Conv2D(16, (3, 3), activation='relu', padding='same')(c2)   ## c2: output tensor of the second Convolutional Layer
# p2 = MaxPooling2D((2, 2))(c2)  ## p2: output tensor of Max Pooling layers
#
# c3 = Conv2D(32, (3, 3), activation='relu', padding='same')(p2)
# c3 = Conv2D(32, (3, 3), activation='relu', padding='same')(c3)  ## c3: output tensor of the thirth Convolutional Layer
# p3 = MaxPooling2D((2, 2))(c3)  ## p3: output tensor of Max Pooling layers
#
# c4 = Conv2D(64, (3, 3), activation='relu', padding='same')(p3)
# c4 = Conv2D(64, (3, 3), activation='relu', padding='same')(c4)   ## c4: output tensor of the fourth Convolutional Layer
# p4 = MaxPooling2D((2,2))(c4)  ## p4: output tensor of Max Pooling Layers
#
# c5 = Conv2D(128, (3, 3), activation='relu', padding='same')(p4)
# c5 = Conv2D(128, (3, 3), activation='relu', padding='same')(c5)  ## c5: output tensor of the fifth Convolutional layer
#
# u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c5)  ## u6: output tensor of up-sampling
# u6 = concatenate([u6, c4])      ###  (transposed convolutional) layers
# c6 = Conv2D(64, (3, 3), activation='relu', padding='same')(u6)
# c6 = Conv2D(64, (3, 3), activation='relu', padding='same')(c6)  ## output tensor of Convolutional Layers
#
# u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c6)  ### output tensors of up-sampling layers
# u7 = concatenate([u7, c3])
# c7 = Conv2D(32, (3, 3), activation='relu', padding='same')(u7)
# c7 = Conv2D(32, (3, 3), activation='relu', padding='same')(c7)  ## output tensors of Convolutional layeers
#
# u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c7)  ## output tensors of up-sampling layers
# u8 = concatenate([u8, c2])
# c8 = Conv2D(16, (3, 3), activation='relu', padding='same')(u8)  ### output tensors of Convolutional layers
# c8 = Conv2D(16, (3, 3), activation='relu', padding='same')(c8)
#
# u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same')(c8)  ## output tensors of up-sampling layers
# u9 = concatenate([u9, c1])
# c9 = Conv2D(8, (3, 3), activation='relu', padding='same')(u9)
# c9 = Conv2D(8, (3, 3), activation='relu', padding='same')(c9) ## output tensor of the last convolutional layers
#
# outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

################################################################
######        Training    ######################################
################################################################

## The model is compiled with "Adam" optimizer

#model = Model(inputs=[inputs], outputs=[outputs])

## The binary crossentropy is used as loss function due that our variable is binary (exist or not exist)
#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])
#model.summary()

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

## Early stopping is used if the validation loss does not improve for 10 continues epochs
#earlystopper = EarlyStopping(patience=5, verbose=1)
## Save the weights only if there is improvement in validation loss
#checkpointer = ModelCheckpoint('model-tgs-salt-1.h5', verbose=1, save_best_only=True)

#results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=8, epochs=30,
#                    callbacks=[earlystopper, checkpointer])







