import os
import cv2
import numpy as np
#import tensorflow as tf


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
                   'N:/cuailp/license_plate_py/test_img/']
mainPath = ''
imgMatrix = []

whois = str.upper(input('are you Sree (S) or Anthony (A)?'))
if whois == 'S':
    mainPath = dirPathListings[0]
elif whois == 'A':
    mainPath = dirPathListings[1]


def import_matrix():
    for filename in os.listdir(mainPath):
        if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):
            currentPath = mainPath + filename
            print('Working with: ', colour.BOLD + colour.BLUE + filename + colour.END, 'which has path: ',
                colour.BOLD + colour.BLUE + currentPath + colour.END)
            print(cv2.imread(currentPath), '\n')
            imgMatrix.append(cv2.imread(currentPath))


def show_all(thisMatrix):
    count = 0
    for i in range(len(thisMatrix) - 1):
        windowName = 'Window' + str(i)
        cv2.namedWindow(windowName, i)
        cv2.imshow(windowName, thisMatrix[i])
        cv2.waitKey(0)
        cv2.destroyWindow(windowName)
        count += 1
    print('there are ', colour.BLUE + str(count) + colour.END, ' photos')


def twoToOneDimension(arr):
    flatter_arr = []
# Input: arr - a two dimensional array
# Returns:vector - a one dimensional contiguous (as given by ravel()function) array
def twoToOneDimension(arr):
    flat_arr = arr.ravel()
    # convert it to a matrix
    return flat_arr


# <<<<<<< HEAD
def principal_component_analysis(arr):
    for a in range(11):
        flatter_arr.append(arr.ravel())
    return flatter_arr
def principal_component_analysis(arr):
    vector = []
    mean = []
    for a in range (0, arr.size - 1):
        vector.append(twoToOneDimension(arr[a]))
        mean.append(int(np.mean(vector[a])))
    print('mean', mean)
    diff = []
    for a in range(0, vector.size - 1):
        diff.append(vector[a] - np.mean(mean))

    # for a in range(1, 11):
    #     diff_vector[a] = vector[a] - mean
    # cov_matrix = np.cov(vector)
    # return diff_vector
    
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

    show_all(numpy_matrix)
    # numpy_vector = twoToOneDimension(numpy_matrix)
    # print('vectorized', numpy_vector)
    # print('vector size', (numpy_vector.shape))
    #print('diff vector', principal_component_analysis(numpy_matrix))
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
