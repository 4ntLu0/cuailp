import os
import sys
import tensorflow as tf
import numpy as np

from model import yolov3
from utils.misc_utils import parseAnchors, loadWeights
import args


num_class = 80
img_size = 416
print(args.weight_path)
print(args.save_path)

