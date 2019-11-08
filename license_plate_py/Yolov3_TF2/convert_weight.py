import os
import sys
import tensorflow as tf
import numpy as np

from model import yolov3
from utils.misc_utils import parseAnchors, loadWeights
from args import weight_path, save_path

num_class = 80
img_size = 416
print(weight_path)
print(save_path)

