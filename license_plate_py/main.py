import cv2
import os
import numpy as np
import tensorflow

def importImgList():
    """Imports all the images into a list.
    """
    inputDir = input('Please enter image folder: ')
    imgList = []
    for filename in os.listdir(inputDir):
        currentPath = inputDir + filename
        imgList.append(cv2.imread(currentPath))
    return imgList

if __name__ == '__main__':
    imgMatrix = np.array(importImgList()) # imports images as a numpy array
    # sree put your squashing in here