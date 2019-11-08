import os
import sys
import tensorflow as tf
import numpy as np

from model import yolov3
from utils.misc_utils import parseAnchors, loadWeights
from args import getWeights, getSavePath, getAnchors

num_class = 80
img_size = 416
weight_path = getWeights()
save_path = getSavePath()
anchors = getAnchors()

model = yolov3(80, anchors)
inputs = tf.placeholder(tf.float32, [1, img_size, img_size, 3])

with tf.variable_scope('yolov3'):
    feature_map = model.forward(inputs)

saver = tf.train.saver(var_list=tf.global_variables(scope='yolov3'))

def forward(variables, path):
	return loadWeights(variables, path)

sess = forward(tf.global_variables(scope='yolov3'), weight_path)
saver.save(sess, save_path=save_path)
print('Tensorflow model checkpoint has been saved to {}'.format(save_path))