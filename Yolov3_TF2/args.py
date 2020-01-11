# coding: utf-8
# this file contains the parameters used in train.py

import math
from utils.misc_utils import parseAnchors, readClassNames #double check import

# some random paths
dirPathListings = ['C:/Users/User/Documents/GitHub/cuailp/license_plate_py/Yolov3_TF2/',
                   'N:/cuailp/license_plate_py/Yolov3_TF2/',
                   'D:/cuailp/license_plate_py/Yolov3_TF2/']
main_path = dirPathListings[2]  # sets the main path that you will be using
train_file = main_path + 'data/my_data/train.txt'  # path of the training txt file
val_file = main_path + 'data/my_data/val.txt'  # path of the validation txt file
restore_path = main_path + 'data/darknet_weights/yolov3.ckpt'  # path of weights to restore
save_dir = main_path + 'checkpoint/'  # directory of the weights to save
log_dir = main_path + 'data/logs/'  # directory to store tensorboard log files
progress_log_path = main_path + 'data/progress.log'  # path to record the training progress
anchor_path = main_path + 'data/yolo_anchors.txt'  # the path of the anchor txt file
class_name_path = main_path + 'data/coco.names'  # path of the class names
weights_path = main_path + 'data/darknet_weights/yolov3.weights'
save_path = main_path + 'data/darknet_weights/yolov3.ckpt'



# Training related numbers
batch_size = 6
img_size = [416, 416]  # images will be resized to [x,y] before feeding to network
letterbox_resize = True  # keep original aspect ratio?
total_epoches = 100
train_evaluation_step = 100  # evaluate on the training batch after ? steps
val_evaluation_epoch = 2  # evaluate on the whole validation dataset after + epochs
save_epoch = 10  # save the model after ? epochs
batch_norm_decay = 0.99  # decay in bn ops
weight_decay = 5 * 10 ** (-4)  # 12 weight decay
global_step = 0  # used when resuming training

# tf.data params
num_threads = 10  # number of threads for image processing in tf.data pipeline.
prefetech_buffer = 5  # prefetech buffer used in tf.data pipeline

# Learning rate and optimizer
optimizer_name = 'momentum'  # chosen from [sgd, momentum, adam, rmsprop]
save_optimizer = True  # save the optimizer params into the checkpoint?
learning_rate_init = 1 * 10 ** (-4)
lr_type = 'piecewise'  # chosen from [fixed, exponential, cosine_decay, cosine_decay, cosine_decay_restart, piecewise]
lr_decay_epoch = 5  # epochs after which learning rate decays. Int or float. used when chosen 'exponential' and 'cosine_decay_restart' lr_type
lr_decay_factor = 0.96  # the learning rate decay factor. Used when chosen 'exponential'
lr_lower_bound = 1 * 10 ** (-6)  # the minimum learning rate
pw_boundaries = [30, 50]  # epoch based boundaries, only for piecewise
pw_values = [learning_rate_init, 3 * 10 ** (-5), 1 * 10 ** (-5)]

# load and finetune
"""
choose the parts you want to restore the weights. List form.
restore_include: None, restore_exclude: None  => restore the whole model
restore_include: None, restore_exclude: scope  => restore the whole model except `scope`
restore_include: scope1, restore_exclude: scope2  => if scope1 contains scope2, restore scope1 and not restore scope2 (scope1 - scope2)
choice 1: only restore the darknet body
restore_include = ['yolov3/darknet53_body']
restore_exclude = None
choice 2: restore all layers except the last 3 conv2d layers in 3 scale
"""
restore_include = None
restore_exclude = ['yolov3/yolov3_head/Conv-14', 'yolov3/yolov3_head/Conv6', 'yolov3/yolov3_head/Conv_22']
# choose the parts you want to finetune. list form.
# set to none to train the whole model.
update_part = ['yolov3/yolov3_head']

# other training strategies
multi_scale_train = True  # whether to apply multi-scale training strategy, image size varies from [320, 320] to [640, 640] by default
use_label_smooth = True  # whether to use class label smoothing strategy
use_focal_loss = True  # whether to apply focal loss on the conf loss.
use_mix_up = True  # whether to use mix up data augmentation strategy
use_warm_up = True  # whether to use warm up strategy to prevent gradient exploding
warm_up_epoch = 3  # warm up training epoches. Set to a larger value if gradient explodes

# some constants in validation
# nms
nms_threshold = 0.45  # iou threshold in nms operation
score_threshold = 0.01  # threshold of the probability of the classes in nms operation, i.e. score = pred_confs * prod_probs. set lower for higher recall
# mAP eval
eval_threshold = 0.5  # the iou threshold applied in mAP evaluation.
use_voc_07_metric = False  # whether to use voc 2007 eval metric (11-point metric)

anchors = parseAnchors(anchor_path)
classes = readClassNames(class_name_path)
class_num = len(classes)
train_img_cnt = len(open(train_file, 'r').readlines())
val_img_cnt = len(open(val_file, 'r').readlines())
train_batch_num = int(math.ceil(float(train_img_cnt) / batch_size))

lr_decay_freq = int(train_batch_num * lr_decay_epoch)
pw_boundaries = [float(i) * train_batch_num + global_step for i in pw_boundaries]

def getWeights():
    return weights_path
def getSavePath():
    return save_path
def getAnchors():
    return anchors