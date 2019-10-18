import cv2
import matplotlib.pyplot as plt
import os
from pathlib import Path

class colour:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

img_matrix = []
mainPath = 'F:/Cuyow/cuailp/license_plate_py/test_img/'
for filename in os.listdir("F:/Cuyow/cuailp/license_plate_py/test_img"):
    currentPath = mainPath + filename
    print('Working with: ', colour.BOLD + colour.BLUE + filename + colour.END, 'which has path: ', colour.BOLD + colour.BLUE + currentPath + colour.END)
    print(cv2.imread(currentPath), '\n')
    img_matrix.append(cv2.imread(filename))

    # Why is it returning none type for the last image?
'''
    print(filename)
    cv2.namedWindow('Window')
    cv2.imshow('Window', cv2.imread(filename))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''

exit()

for i in range(len(img_matrix) - 1):
    windowName = 'Window' + str(i)
    cv2.namedWindow(windowName, i)
    cv2.imshow(windowName, img_matrix[i])
    cv2.waitKey(0)


testimg = cv2.imread('F:\\Cuyow\\cuailp\\license_plate_py\\testing_imgs\\Testing.jpg') #reads image into testimg
#print(testimg)

# Attempt to display using cv2 (single image)
"""
cv2.namedWindow("Window")
cv2.imshow("Window", testimg)
cv2.waitKey(0)
"""

print(img_matrix)