# -*- coding: utf-8 -*-
import tensorflow as tf
from skimage import io,transform
import os
from datareader import datareader
from network import BuildNet
import matplotlib.pyplot as plt
import numpy as np

test_img = './tid2008/distorted_images/i01_04_4.bmp'
img = io.imread(test_img)
img = transform.resize(img, (64, 64))

''' network build'''
# Input and target placeholders
inputs_ = tf.placeholder(tf.float32, (None, 64,64,3), name="input")

# build network
print('Building network')
pre = BuildNet(inputs_)

# load model
loader = tf.train.Saver() 

with tf.Session() as sess:
    # restore model
    model_file=tf.train.latest_checkpoint('./snapshots/')
    loader.restore(sess, model_file) 
    # predict
    pre_ = sess.run([pre], feed_dict={inputs_:[img]})

# show results
plt.subplot(1,2,1)
plt.imshow(img)
plt.subplot(1,2,2)
plt.imshow(np.squeeze(pre_[0])+img)
plt.show()

