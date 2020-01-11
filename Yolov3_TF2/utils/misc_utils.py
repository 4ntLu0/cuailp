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

def loadWeights(var_list, weights_file):
    """
    Loads and converts pre-trained weights.
    :param var_list: List of network variables
    :param weights_file: Name of th ebinary file
    :return:
    """
    with open(weights_file, 'rb') as fp:
        np.fromfile(fp, dtype = np.int32, count = 5)
        weights = np.fromfile(fp, dtype = np.float32)

        ptr = 0
        i = 0
        assign_ops = []
        while i < len(var_list) -1:
            var1 = var_list[i]
            var2 = var_list[i +1]
            # do something only if we process conv layer
            if 'Conv' in var1.name.split('/')[-2]:
                # check type of next layer
                if 'BatchNorm' in var2.name.split('/')[-2]:
                    # load batch norm params
                    gamma, beta, mean, var = var_list[i + 1: i + 5]
                    batch_norm_vars = [beta, gamma, mean, var]
                    for var in batch_norm_vars:
                        shape = var.shape.as_list()
                        num_params = np.prod(shape)
                        var_weights = weights[ptr:ptr + num_params].reshape(shape)
                        ptr += num_params
                        assign_ops.append(tf.assign(var,var_weights, validate_shape = True))
                    #move pointer by 4, because 4 vals
                    i += 4
                elif 'Conv' in var2.name.split('/')[-2]:
                    #load biases
                    bias = var2
                    bias_shape = bias.shape.as_list()
                    bias_params = np.prod(bias_shape)
                    bias_weights = weights[ptr: ptr + bias_params].reshape(bias.shape)
                    #we loaded 1 val
                    i += 1
            shape = var1.shape.as_list()
            num_params = np.prod(shape)
            var_weights = weights[ptr: ptr + num_params].reshape((shape[3], shape[2], shape[0], shape[1]))
            #transpose to column-major
            var_weights = np.transpose(var_weights,(2,3,1,0))
            ptr += num_params
            assign_ops.append( tf.assign(var1, var_weights, validate_shape = True))
            i += 1

    return assign_ops


def readClassNames(class_name_path): 
    '''
    --- Inputs ---
    class_name_path - string representing the path of the class file

    -- Use --
    Read in all the class name files that are needed

    --- Algorithm ---
    Open file class_name_path
    Read in each value as a data object, not a string
    Key each id to its corresponding class name in the dictionary:
        i.e. names = {"id" : "name", "1", "class_name"}

    --- Returns ---
    names - dictionary containing a series of class names
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
        weights = np.fromfile(fp, dtype = np.float32)

    ptr = 0
    i = 0
    assign_ops = []
    while i < len(var_list) - 1:
        var1 = var_list[i]
        var2 = var_list[i + 1]
        if 'Conv' in var1.name.split('/')[-2]:
            # check type of next layer
            if 'BatchNorm' in var2.name.split('/')[-2]:
                # load batch norm params
                gamma, beta, mean, var = var_list[i + 1:i + 5]
                batch_norm_vars = [beta, gamma, mean, var]
                for var in batch_norm_vars:
                    shape = var.shape.as_list()
                    num_params = np.prod(shape)
                    var_weights = weights[ptr:ptr + num_params].reshape(shape)
                    ptr += num_params
                    assign_ops.append(tf.assign(var, var_weights, validate_shape=True))
                # we move the pointer by 4, because we loaded 4 variables
                i += 4
            elif 'Conv' in var2.name.split('/')[-2]:
                # load biases
                bias = var2
                bias_shape = bias.shape.as_list()
                bias_params = np.prod(bias_shape)
                bias_weights = weights[ptr:ptr +
                                           bias_params].reshape(bias_shape)
                ptr += bias_params
                assign_ops.append(tf.assign(bias, bias_weights, validate_shape=True))
                # we loaded 1 variable
                i += 1
            # we can load weights of conv layer
            shape = var1.shape.as_list()
            num_params = np.prod(shape)

            var_weights = weights[ptr:ptr + num_params].reshape(
                (shape[3], shape[2], shape[0], shape[1]))
            # remember to transpose to column-major
            var_weights = np.transpose(var_weights, (2, 3, 1, 0))
            ptr += num_params
            assign_ops.append(
                tf.assign(var1, var_weights, validate_shape=True))
            i += 1

        return assign_ops