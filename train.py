# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 17:52:57 2018

@author: admin
"""
import numpy as np
import tensorflow as tf
#import matplotlib; matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
from datareader import datareader
from network import BuildNet
import os

''' parameters setting '''
# dataset path
datapath1 = './tid2008/reference_images/'
datapath2 = './tid2008/distorted_images/'
show_results = 0
# learning rate
learning_rate = 0.01
# training epochs
epochs = 200
# batch size
batch_size = 167

'''data read'''
print('Reading data')
x,y = datareader(datapath1,datapath2)
input_x = np.zeros((len(x),64,64,3))
input_y = np.zeros((len(y),64,64,3))
for i in range(len(x)):
    input_x[i,:,:,:] = x[i]
    input_y[i,:,:,:] = y[i]
print('data set: ',input_x.shape)
print('label set: ',input_y.shape)

if(show_results):
    plt.subplot(1,2,1)
    plt.imshow(x[18])
    plt.subplot(1,2,2)
    plt.imshow(y[18])
    plt.show()

''' network build'''
# Input and target placeholders
inputs_ = tf.placeholder(tf.float32, (None, 64,64,3), name="input")
targets_ = tf.placeholder(tf.float32, (None, 64,64,3), name="target")

# build network
print('Building network')
pre = BuildNet(inputs_)
pre = pre + inputs_
# Pass logits through sigmoid and calculate the cross-entropy loss
loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets_, logits=pre)
# Get cost and define the optimizer
cost = tf.reduce_mean(loss)
# optimization using adam
opt = tf.train.AdamOptimizer(learning_rate).minimize(cost)
# tensorboard
tf.summary.scalar('loss', cost)
merged = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter('./train', graph=tf.get_default_graph())


'''build session'''
sess = tf.Session()
# initialization
sess.run(tf.global_variables_initializer())
# for each epoch
input_x = np.zeros((batch_size,64,64,3))
input_y = np.zeros((batch_size,64,64,3))
it = 0
for e in range(epochs):
    # for each batch
    for ii in range(len(x)//batch_size):
        # input data and running iter
        batch_cost, _,merged_ = sess.run([cost, opt,merged], feed_dict={inputs_: input_x[ii:ii+batch_size],targets_: input_y[ii:ii+batch_size]})
        # write to tensorboart
        summary_writer.add_summary(merged_, it)
        it = it +1
        print("Epoch: {}/{}...".format(e+1, epochs),
              "Training loss: {:.4f}".format(batch_cost))

if not os.path.exists('./snapshots'):
        os.system("mkdir ./snapshots")
saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)  
saver.save(sess, "./snapshots/AutoEncode_model.ckpt", global_step=0)  
plt.show()

