import tensorflow as tf
import tf_slim as slim

from utils.layer_utils import conv2d, darknet53Body, yoloBlock, upsampleLayer  # double check import


class yolov3(object):

    def __init__(self, class_num, anchors, use_label_smooth=False, use_focal_loss=False, batch_norm_decay=0.999,
                 weight_decay=5 * 10 ** (-4), use_static_shape=True):
        # self.anchors = [[10,13],[16,30],[33,23],[30,61],[62,45],[59,119],[116,90],[156,198],[373,326]]
        self.class_num = class_num
        self.anchors = anchors
        self.batch_norm_decay = batch_norm_decay
        self.use_label_smooth = use_label_smooth
        self.use_focal_loss = use_focal_loss
        self.weight_decay = weight_decay
        '''inference speed optimization
        if 'use_static_shape' is True, use tensor.get_shape(), otherwise use tf.shape(tensor)
        static shape is slightly faster'''
        self.use_static_shape = use_static_shape

    def forward(self, inputs, is_training=False, reuse=False):
        'the input imG_size, form: [height, weight] will be used later'
        self.img_size = tf.shape(inputs)[1:3]
        # set batch norm params
        batch_norm_params = {
            'decay': self.batch_norm_decay,
            'epsilon': 1 * 10 ** (-5),
            'scale': True,
            'is_training': is_training,
            'fused': None,  # Use fused batch norm if possible ?? SREE HELP
        }


"I'm stopping here because I want to make sure before continue"
