import numpy as np
import tensorflow as tf
import random

'something about summary_pb2 we will have to add this in later when we get there'

def parseAnchors(anchor_path):
    '''
    parse anchors,
    :param anchor_path:
    :return: shape [N, 2], dtype float 32
    '''
    anchors = np.reshape(np.asarray(open(anchor_path, 'r').read().split(','), np.float32), [-1,2])
    return anchors

def readClassNames(class_name_path): #sree I actually don't know how this works
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

def load_weights(var_list, weights_file):
    '''
    loads and cornverts pre trained weights
    :param var_list: list of network variables
    :param weights_file: name of the binary file
    :return:
    '''
    with open(weights_file, 'rb') as fp:
        np.fromfile(fp, dtype=np.float32)

    ptr = 0
    i = 0
    assign_ops = []
    while i < len(var_list) - 1:
        var1 = var_list[i]
        var2 = var_list[i + 1]
        if 'Conv' in var1.name.split('/')[-2]