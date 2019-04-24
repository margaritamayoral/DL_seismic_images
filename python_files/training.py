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

from import_data import im_chan, im_width, im_height, train_ids, path_train
from U_Net import inputs, outputs

##Get and resize train images and masks

X_train = np.zeros((len(train_ids), im_height, im_width, im_chan), dtype=np.uint8)
Y_train = np.zeros((len(train_ids), im_height, im_width, 1), dtype=np.bool)
print('Getting and resizing train images and masks')
sys.stdout.flush()
for n, id_ in tqdm_notebook(enumerate(train_ids), total=len(train_ids)):
    path = path_train
    img = load_img(path + '/images/' + id_)
    x = img_to_array(img)[:, :, 1]
    x = resize(x, (128, 128, 1), mode='constant', preserve_range=True)
    X_train[n] = x
    mask = img_to_array(load_img(path + '/masks/' + id_))[:, :, 1]
    Y_train[n] = resize(mask, (128, 128, 1), mode='constant', preserve_range=True)

print('Done!')

### Check if training data looks allright

ix = random.randint(0, len(train_ids))
ax2 = plt.figure()
plt.imshow(np.dstack((X_train[ix], X_train[ix], X_train[ix])))
plt.savefig('../output/showing_train_data_x.png')

tmp = np.squeeze(Y_train[ix]).astype(np.float32)
plt.imshow(np.dstack((tmp,tmp,tmp)))
plt.savefig('../output/showing_train_data_y.png')


###############################################################
######        Training    ######################################
################################################################

## The model is compiled with "Adam" optimizer

model = Model(inputs=[inputs], outputs=[outputs])

## The binary crossentropy is used as loss function due that our variable is binary (exist or not exist)


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])
model.summary()

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

## Early stopping is used if the validation loss does not improve for 10 continues epochs

earlystopper = EarlyStopping(patience=5, verbose=1)

## Save the weights only if there is improvement in validation loss

checkpointer = ModelCheckpoint('model-tgs-salt-1.h5', verbose=1, save_best_only=True)