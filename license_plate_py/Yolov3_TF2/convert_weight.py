import os
import sys
import tensorflow as tf
import numpy as np

from model import yolov3
from utils.misc_utils import parseAnchors, loadWeights
from args import getWeights, getSavePath, getAnchors

num_class = 80
img_size = 416
weight_paths = getWeights()
save_path = getSavePath()
anchors = getAnchors()

model = yolov3(80, anchors)
with tf.session() as sess:
    