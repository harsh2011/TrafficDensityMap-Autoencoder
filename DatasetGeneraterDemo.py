from darkflow.net.build import TFNet
import numpy as np
import cv2
import json 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time


# YOLO weights and adjust the threshold value of t
# please download the YOLO weights
options = {"model": "cfg/yolo.cfg", "load": "bin/yolo.weights", "threshold": 0.5}

tfnet = TFNet(options)


def getdensity(image):
    '''
    This function pass the frame in YOLO. YOLO output the object 
    detected in the frame and its position. Extracting only the vehicles
    from the data and generate a image of same size wth 0s and 1s.
    Where the Vehicle is detected pixel value should come 1s and 
    others comes 0s 
    '''
    imgcv =np.array(image)
    height, width = imgcv.shape[:2]
    print(height , ":", width)
    
    result = tfnet.return_predict(imgcv)
    

    # object that are extracted from the data
    motorbike=0;
    bicycle=0;
    car=0;
    bus=0;
    truck=0;

    # initalizing the output array with zeros
    a = np.zeros((height,width))

    for value in result:
        if(value["label"]=='car'):
            car = car+1
            ones = np.ones((value['bottomright']['y']-value['topleft']['y'],value['bottomright']['x']-value['topleft']['x']))
            bounding_box = [value['topleft']['x']-2,value['topleft']['y']-2,value['bottomright']['x']-value['topleft']['x']-2,value['bottomright']['y']-value['topleft']['y']-2]
            
            # making the detect vechile position image pixel value 1
            a[bounding_box[1]:bounding_box[1] + bounding_box[3], bounding_box[0]:bounding_box[0] + bounding_box[2]] = 1
        if(value["label"]=='bus'):
            bus = bus+1
            ones = np.ones((value['bottomright']['y']-value['topleft']['y'],value['bottomright']['x']-value['topleft']['x']))
            bounding_box = [value['topleft']['x']-2,value['topleft']['y']-2,value['bottomright']['x']-value['topleft']['x']-2,value['bottomright']['y']-value['topleft']['y']-2]
            
            # making the detect vechile position image pixel value 1
            a[bounding_box[1]:bounding_box[1] + bounding_box[3], bounding_box[0]:bounding_box[0] + bounding_box[2]] = 1
        if(value["label"]=='truck'):
            truck = truck+1
            ones = np.ones((value['bottomright']['y']-value['topleft']['y'],value['bottomright']['x']-value['topleft']['x']))
            bounding_box = [value['topleft']['x']-2,value['topleft']['y']-2,value['bottomright']['x']-value['topleft']['x']-2,value['bottomright']['y']-value['topleft']['y']-2]

            # making the detect vechile position image pixel value 1
            a[bounding_box[1]:bounding_box[1] + bounding_box[3], bounding_box[0]:bounding_box[0] + bounding_box[2]] = 1
        if(value["label"]=='motorbike'):
            motorbike = motorbike+1
            ones = np.ones((value['bottomright']['y']-value['topleft']['y'],value['bottomright']['x']-value['topleft']['x']))
            bounding_box = [value['topleft']['x']-2,value['topleft']['y']-2,value['bottomright']['x']-value['topleft']['x']-2,value['bottomright']['y']-value['topleft']['y']-2]

            # making the detect vechile position image pixel value 1 
            a[bounding_box[1]:bounding_box[1] + bounding_box[3], bounding_box[0]:bounding_box[0] + bounding_box[2]] = 1
        if(value["label"]=='bicycle'):
            bicycle = bicycle+1
            ones = np.ones((value['bottomright']['y']-value['topleft']['y'],value['bottomright']['x']-value['topleft']['x']))
            bounding_box = [value['topleft']['x']-2,value['topleft']['y']-2,value['bottomright']['x']-value['topleft']['x']-2,value['bottomright']['y']-value['topleft']['y']-2]

            # making the detect vechile position image pixel value 1
            a[bounding_box[1]:bounding_box[1] + bounding_box[3], bounding_box[0]:bounding_box[0] + bounding_box[2]] = 1

    # return the 0s and 1s image to the main function
    return a


image = cv2.imread('sample1.png')

a = getdensity(image)
a = a*255

cv2.imwrite('output.png',a)

cv2.waitKey(0)
cv2.destroyAllwindow()