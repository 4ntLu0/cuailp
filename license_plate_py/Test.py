import cv2
import matplotlib.pyplot as plt
import os

img_matrix = []
for filename in os.listdir("F:\\Cuyow\\cuailp\\license_plate_py\\testing_imgs"):
    img_matrix.append(cv2.imread(filename))
    print(cv2.imread(filename))

    # Why is it returning none type for the last image?

    print(filename)
    cv2.namedWindow('Window')
    cv2.imshow('Window', cv2.imread(filename))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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