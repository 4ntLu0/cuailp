import os
import opencv as cv2
import numpy as np
from darknetpy.detector import detector

cfgFile = './cfg/yolov3.cfg'
weightFile = './weights/yolov3.weights'
namesfile = 'data/coco.names'

m = Darknet(cfgFile)

classNames = loadClassNames(namesfile)