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



mainPath = 'C:/Users/User/Documents/GitHub/cuailp/license_plate_py/test_img/'
img_matrix = []


def import_matrix():
    for filename in os.listdir(mainPath):
        currentPath = mainPath + filename
        print('Working with: ', colour.BOLD + colour.BLUE + filename + colour.END, 'which has path: ',
              colour.BOLD + colour.BLUE + currentPath + colour.END)
        print(cv2.imread(currentPath), '\n')
        img_matrix.append(cv2.imread(currentPath))




def show_all():
    for i in range(len(img_matrix) - 1):
        windowName = 'Window' + str(i)
        cv2.namedWindow(windowName, i)
        cv2.imshow(windowName, img_matrix[i])
        cv2.waitKey(0)


def twoToOneDimension(arr):
    flatter_arr = []
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
    numpy_matrix = np.array(img_matrix)
    print('printing numpy matrix', numpy_matrix)
   # numpy_vector = twoToOneDimension(numpy_matrix)
   # print('vectorized', numpy_vector)
   # print('vector size', (numpy_vector.shape))
    print('raveled', principal_component_analysis(numpy_matrix))
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
    #show_all()
