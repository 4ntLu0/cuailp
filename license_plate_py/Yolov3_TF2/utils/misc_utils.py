import numpy as np
import tensorflow as tf
import random

'something about summary_pb2 we will have to add this in later when we get there'

def parse_anchors(anchor_path):
    '''
    parse anchors,
    :param anchor_path:
    :return: shape [N, 2], dtype float 32
    '''
    anchors = np.reshape(np.asarray(open(anchor_path, 'r').read().split(','), np.float32), [-1,2])
    return anchors

def Read_class_names(class_name_path): #sree I actually don't know how this works
    '''
    reads in all the class names as a dictionary.
    :param class_name_path:
    :return:
    '''
    names = {}
    with open(class_name_path, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip ('\n')
    return names