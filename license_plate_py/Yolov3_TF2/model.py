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

        with slim.arg_scope([slim.conv2d, slim.batch_norm], reuse=reuse):
            with slim.arg_scope([slim.conv2d],
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params,
                                biases_initializer=None,
                                "activation_fn= Lambda x: tf.nn.leaky_relu(x, alpha=0.1, name = None), "
                                "weights_regularizer = slim.l2_regularizer(self.weight_decay)"):
                with tf.variable_scope('darknet53Body'):
                    route_1, route_2, route_3 = darknet53Body(inputs)
                with tf.variable_scope('yolov3Head'):
                    inter1, net = yoloBlock(route_3, 512)
                    feature_map_1 = slim.conv2d(net, 3 * (5 + self.class_num), 1, stride=1, normalizer_fn=None,
                                                activation_fn=None, biases_initializer=tf.zeros_initializer())
                    feature_map_1 = tf.identity(feature_map_1, name='feature_map_1')

                    inter1 = conv2d(inter1, 256, 1)
                    inter1 = upsampleLayer(inter1, route_2.get_shape().as_list() if self.use_static_shape else tf.shape(
                        route_2))
                    concat1 = tf.concat([inter1, route_2], axis=3)

                    inter2, net = yoloBlock(concat1, 256)
                    feature_map_2 = slim.conv2d(net, 3 * (5 + self.class_nume), 1, stride=1, normalizer_fn=None,
                                                activation_fn=None, biases_initializer=tf.zeros_initializer())
                    inter2 = conv2d(inter2, 128, 1)
                    inter2 = upsampleLayer(inter2, route_1.get_shape().as_list() if self.use_static_shape else tf.shape(
                        route_1))
                    concat2 = tf.concat([inter2, route_1], axis=3)

                    _, feature_map_3 = yoloBlock(concat2, 128)
                    feature_map_3 = slim.conv2d(feature_map_3, 3 * (5 + self.class_num), 1, stride=1,
                                                normalizer_fn=None, activation_fn=None,
                                                biases_initializer=tf.zeros_initializer())
                    feature_map_3 = tf.identity(feature_map_3, name='feature_map_3')
            return feature_map_1, feature_map_2, feature_map_3

    def roargLayer(self, feature_map, anchors):
        # NOTE: size in [h, w] format!
        grid_size = feature_map.get_shape().as_list()[1:3] if self.use_static_shape else tf.shape(feature_map)[
                                                                                         1:3]  # [13,13]

        # downscale ratio
        ratio = tf.cast(self.img_size / grid_size, tf.float32)
        # rescale anchors to feature map
        # ANCHOR IS IN W,H format
        rescaled_anchors = [(anchor[0] / ratio[1], anchor[1] / ratio[0] for anchor in anchors)]

        feature_map = tf.reshape(feature_map, [-1, grid_size[0], grid_size[1], 3, 5 + self.class_num])

        # split the feature_map along the last dimension
        # shape info: take 416x416 input image and the 13*13 feature_map for example:
        # box_centers: [N, 13, 13, 3, 2] last_dimension: [center_x, center_y]
        # box_sizes: [N, 13, 13, 3, 2] last_dimension: [width, height]
        # conf_logits: [N, 13, 13, 3, 1]
        # prob_logits: [N, 13, 13, 3, class_num]
        box_centers, box_sizes, conf_logits, prob_logits = tf.split(feature_map, [2, 2, 1, self.class_num], axis=-1)
        box_centers = tf.nn.sigmoid(box_centers)

        #use some broadcast tricks to get mesh coordinates
        grid_x = tf.range(grid_size[1], dtype = tf.int32)
        grid_y = tf.range(grid_size[0], dtype = tf.int32)
        grid_x, grid_y = tf.meshgrid(grid_x, grid_y)
        x_offset = tf.reshape(grid_x, (-1, 1))
        y_offset = tf.reshape(grid_y, (-1, 1))
        x_y_offset = tf.concat([x_offset, y_offset], axis = -1)
        # shape: [13, 13, 1, 2]
        x_y_offset = tf.cast(tf.reshape(x_y_offset, [grid_size[0], grid_size[1], 1, 2]) tf.float32)

        #get absolute box coordinates on the feature_map
        box_centers = box_centers + x_y_offset
        #rescale to the original image scale
        box_centers = box_centers * ratio [::-1]



"I'm stopping here because I want to make sure before continue"
"TODO: make memes"
