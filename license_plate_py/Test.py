import os
import cv2
import numpy as np
from test2 import printer #very interesting that this works
import tensorflow as tf


class colour:
    BLACK = '\033[97m'
    GREY = '\033[37m'
    BROWN = '\033[33m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    ITALIC = '\033[3m'
    UNDERLINE = '\033[4m'
    STRIKETHROUGH = '\033[9m'
    END = '\033[0m'


dirPathListings = ['C:/Users/User/Documents/GitHub/cuailp/license_plate_py/test_img/',
                   'N:/cuailp/license_plate_py/test_img/',
                   'D:/cuailp/license_plate_py/test_img/']
mainPath = ''
imgMatrix = []

whois = str.upper(input('are you Sree (S) or Anthony (A)?'))
if whois == 'S':
    mainPath = dirPathListings[0]
elif whois == 'A':
    whois = str.upper(input('laptop (L) or desktop (D)'))
    if whois == 'L':
        mainPath = dirPathListings[2]
    elif whois == 'D':
        mainPath = dirPathListings[1]


def import_matrix():
    for filename in os.listdir(mainPath):
        print(filename, '\n')
        if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):
            currentPath = mainPath + filename
            print('Working with: ', colour.BOLD + colour.BLUE + filename + colour.END, 'which has path: ',
                colour.BOLD + colour.BLUE + currentPath + colour.END)
            print(cv2.imread(currentPath), '\n')
            imgMatrix.append(cv2.imread(currentPath))


def show_all(thisMatrix):
    count = 0
    for i in range(len(thisMatrix)):
        windowName = 'Window' + str(i)
        cv2.namedWindow(windowName, i)
        cv2.imshow(windowName, thisMatrix[i])
        cv2.waitKey(0)
        cv2.destroyWindow(windowName)
        count += 1
    print('there are ', colour.BLUE + str(count) + colour.END, ' photos')


def twoToOneDimension(arr):
    flatter_arr = np.empty(11)
    for a in range(11):
        np.append(flatter_arr, arr.ravel())
    return flatter_arr
def principal_component_analysis(arr):
    vector = np.empty(int(arr.size))
    mean = np.empty(int(arr.size))
    for a in range (0, arr.size - 1):
        np.append(vector,twoToOneDimension(arr[a]))
        np.append(mean, int(np.mean(vector[a])))
    print('mean', mean)
    diff = np.empty(int(vector.size))
    
    for a in range(0, vector.size - 1):
        np.append(diff, vector[a] - np.mean(mean))
    return diff
    
def test_colours():
    print(colour.BLACK + 'black?' + colour.END)

if __name__ == '__main__':
    import_matrix()
    numpy_matrix = np.array(imgMatrix)
    print('printing numpy matrix', numpy_matrix)

   # numpy_vector = twoToOneDimension(numpy_matrix)
   # print('vectorized', numpy_vector)
   # print('vector size', (numpy_vector.shape))
    print('raveled', principal_component_analysis(numpy_matrix))
   # print()

    # show_all(numpy_matrix)
    numpy_vector = twoToOneDimension(numpy_matrix)
    print('vectorized', numpy_vector)
    print('vector size', (numpy_vector.shape))
    print('diff vector', principal_component_analysis(numpy_matrix))
    # print()
    # test_colours()
    '''
    print(len(img_matrix))
    print(len(img_matrix[0]))
    print(len(img_matrix[0][0]))
    print(len(img_matrix[1]))
    print(len(img_matrix[1][0]))
    print(len(img_matrix[2]))
    print(len(img_matrix[2][0]))
    '''
    # show_all()

    #printer()