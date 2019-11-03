# changed a comment up top

import numpy as np
import tensorflow as tf
#import tf_slim as slim

'''
conv2d 
--- Inputs ---
    inputs - a tensor object (4D matrix with rank 4)
    filters - integer representing dimensionality of output space (i.e. the number of output filters in the convolution/ number of neurons in layer)
    kernel size - width and height of a filter mask layer
                - i.e. dimensions of squashing
    stride - int or list of ints (of length, 1 2 or 4);  the size of the sliding window for each dimension of the input
                - if the length is 1 or 2, the batch dimension and channels dimension are set to 1
    padding - string; either "SAME" or "VALID" - referring to padding algorithms:
                "SAME" algorithm: (assumes that stride = 1)
                    Size of output is equal to size of input
                    If a kernel (filter) of size k×k is used, then we choose padding p such that p = (k+1)/2
                        Proof:
                            Consider a nxn input N and a mxm kernel K
                            In order to make the output a nxn matrix, we need to compute the convolutional matrix of the input n times in each direction
                            It follows that the center of the kernel should be placed at each cell of the input matrix
                            We must therefore pad each cell of the input with enough zeroes to make it the center of the kernel
                                The center of the mxm kernel lies at [K]((m+1)/2, (m+1)/2) (assuming m is odd)
                            If we want the center of K to lie on (a,a) for some a <= m, 
                                some horizontal translation must be done to the center of K ((a+1)/2, (a+1)/2)
                            We want to choose p where p + (a+1)/2 = a 
                                                      p = a - (a+1)/2
                                                      p = (a - 1)/2. QED.
                "VALID" algorithm: 
                    Only uses 'valid' input data; that is, no null value (0) padding is added to the data
                    Valid is used when stride is not equal to one, so rightmost-bottommost values that do not fit into filter window
                    So, output will have dimensions of size filterwindow - 1 
                        (so that each element of the filterwindow contains valid input (never inexistent input))
--- Uses ---
    Puts data through a 2 dimensional convoluntional neural network

--- Algorithm ---
    If size of window is 1, then padding algorithm will not drop values,
        So, pass input data through the keras conv2d function
    If size of sliding window is bigger one, padding algorithm will drop values
        So convert input by passing it throuhg _fixed_padding,
        Then pass input through the keras conv2d function
--- Returns ---
 4D tensor with shape: (batch, filters, new_rows, new_cols) if data_format is "channels_first"
'''
def conv2d(inputs , filters, kernel_size, strides=1):

    def _fixed_padding(inputs, kernel_size):
        '''
            --- Inputs ---
            inputs - a tensor object (4D matrix with rank() =4)
            kernel size - integer representing rank of a filter mask layer
                    - i.e. dimensions of squashing
            --- Use ---
            Is called when stride != 1, so adds padding to the input such that input does not drop values under VALID padding algorithm

            --- Algorithm ---
            - Sets total output size to kernel size - 1
            So that each element of the kernel will always correspond to a valid element of the input when convolution occurs
            - Set padding to be added to input as
            - no padding for the 1th dimension
            - padding of half the kernel size before and after for the 2th dimension
            - padding of half the kernel size before and after for the 3th dimension
            - no padding for the last dimension

            --- Returns ---
            padded_inputs - a tensor object with size paddings[D, 0] + tensor.dim_size(D) + paddings[D, 1] for each dimension of the original input
  
        '''
        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg

        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
        return padded_inputs

    if strides > 1:
        inputs = _fixed_padding(inputs, kernel_size)
    return tf.keras.conv2d(inputs, filters, stride=strides, padding=('SAME' if strides == 1 else 'VALID'))


def darknet53Body(inputs):
    '''
        --- Inputs ---
        inputs - a tensor object (4D matrix with rank 4)
        
        --- Use ---
        Runs data through convolutional network over and over again (in order to mimick Darknet53's neural network (latest version))

        --- Algorithm ---
        CNN Logic:
            Input data is sent to each neuron (filter) on the first layer, where the dot product of each pixel in the input data and matrix (kernel)
            is performed to produce a new pixel. 
            The image produced by all neurons/filters in a layer are backpropogated and combined and passed as the input for the next layer of neurons
            
            Using this, we first detect features that can be recognized and interpreted relatively easy (largest features), by passing it through a
            layer with a low neuron density. 
            Then detect increasingly more abstract (and smaller) features, usually present in many previously detected larger features, by passing
            it through layers with increasingly high neuron densities. 
            Finally make a specific classification by combining all the specific features detected by the previous layers in the input data by using 
            a very high density neuron layer. 

            We decrease dimensionality of the resulting feature map each time the data is passed through a layer (from strides = 2), allowing for 
            some compression


        --- Returns ---
        route_1, route_2 and route_3 - Feature maps of 3 different sizes, each detecting features at a different scale
            route_1 has the largest scale, which is downsized through route_2 and route_3
    '''
    def res_block(inputs, filters):
        '''
        --- Inputs ---
        inputs -  a tensor object (4D matrix with rank 4)
        filters - integer representing dimensionality of output space (i.e. the number of output filters in the convolution/ number of neurons in layer)
        
        --- Use ---
        Backpropogate and combine the images produced by filters in two layers

        --- Algorithm ---
        - Pass input through first filter layer with kernel size of 1 so that the output has the same number of pixels as input
        - Pass resultant through next filter layer (multiplication by 2 yeilds next power of 2) with kernel size of 3 in order to reduce number of 
        pixels in output
        - Combine original input with filtered matrix
        
        --- Returns ---
        net - 4D tensor object
        '''
        shortcut = inputs
        net = conv2d(inputs, filters * 1, 1)
        net = conv2d(net, filters * 2, 3)

        net = net + shortcut

        return net

    # first two conv2d layers
    net = conv2d(inputs, 32, 3, strides=1)
    net = conv2d(net, 64, 3, strides=2)

    # res_block * 1
    net = res_block(net, 32)

    net = conv2d(net, 128, 3, strides=2)

    # res_block * 2
    for i in range(2):
        net = res_block(net, 64)

    net = conv2d(net, 256, 3, strides=2)

    # res_block * 8
    for i in range(8):
        net = res_block(net, 128)

    route_1 = net
    net = conv2d(net, 512, 3, strides=2)

    # res_block * 8
    for i in range(8):
        net = res_block(net, 256)

    route_2 = net
    net = conv2d(net, 1024, 3, strides=2)

    # res_block * 4
    for i in range(4):
        net = res_block(net, 512)
    route_3 = net

    return route_1, route_2, route_3

# yoloBlock takes in 
def yoloBlock(inputs, filters): 
    net = conv2d(inputs, filters * 1, 1)
    net = conv2d(net, filters * 2, 3)
    net = conv2d(net, filters * 1, 1)
    net = conv2d(net, filters * 2, 3)
    net = conv2d(net, filters * 1, 1)
    route = net
    net = conv2d(net, filters * 2, 3)
    return route, net


def upsampleLayer(inputs, out_shape):
    new_height, new_width = out_shape[1], out_shape[2]
    # HEIGHT IS THE FIRST
    # TODO: set 'align_corners' as True?
    inputs = tf.image.resize_nearest_neighbor(inputs, (new_height, new_width), name='upsampled')
    return inputs
