import os
import tensorflow as tf
import tensornets as nets
import cv2
import numpy as np
import time

act_start_time = 0
preds = 0

inputs = tf.placeholder(tf.float32, [None, 416, 416, 3])
model = nets.YOLOv3COCO(inputs, nets.Darknet19)

classes = {'0': 'person', '1': 'bicycle', '2': 'car', '3': 'bike', '4': 'airplane', '5': 'bus', '6': 'train',
           '7': 'truck', '8': 'boat', '9': 'traffic light', '10': 'fire hydrant', '12': 'stopsign',
           '13': 'parking meter', '14': 'bench', '15': 'bird', '16': 'cat', '17': 'dog', '18': 'horse', '19': 'sheep',
           '20': 'cow', '26': 'backpack', '72': 'laptop'}
list_of_classes = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 26, 72]

with tf.Session() as sess:
    sess.run(model.pretrained())

cap = cv2.VideoCapture('N:/cuailp/license_plate_py/test_img/city_footage_1')
print(cap)
while(cap.isOpened()):
    ret, frame = cap.read()
    img = cv2.resize(frame,(416,416))
    imge = np.array(img).reshape(-1,416,416,3)
    act_start_time = time.time()
    preds = sess.run(model.preds, {inputs: model.preprocess(imge)})
    print(preds)

ret, frame = cap.read()
img = cv2.resize(frame,(416,416))
imge = np.array(img).reshape(-1,416,416,3)
act_start_time = time.time()
preds = sess.run(model.preds, {inputs: model.preprocess(imge)})
print(preds)

start_time = act_start_time

print("---%s seconds ---" % (time.time() - start_time))
boxes = model.get_boxes(preds, imge.shape[1:3])
cv2.namedWindow('image', cv2.WINDOW_NORMAL)

cv2.resizeWindow('image', 700, 700)
boxes1 = np.array(boxes)
for j in list_of_classes:
    count = 0
    if str(j) in classes:
        lab = classes[str(j)]
    if len(boxes1) != 0:
        for i in range(len(boxes1[j])):
            box = boxes1[j][i]
            if boxes1[j][i] >= .40:
                count += 1

cv2.rectangle(img, (box[0],box[1]), (box[2],box[3]),(0,255,0),3)
cv2.putText(img,lab,(box[0],box[1]), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0,0,255), lineType = cv2.LINE_AA)
print(lab,": ", count)

cv2.imshow('image', img)
if cv2.waitKey(1) & 0xFF == ord('q'):
    pass
    #supposedly going to break the loop
#more code here
