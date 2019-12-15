'''
Created on Dec 11, 2019

@author: HARISH C
'''

import os
import cv2
import numpy as np


def read_image(path):
    img_files = os.listdir(path)
    if (img_files is None or len(img_files) == 0):
        raise FileNotFoundError()
    image = cv2.imread(os.path.sep.join([path, img_files[0]]))
#     cv2.imshow('image',image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
    data = np.empty((len(img_files), image.size), dtype=int)
    for i, file in enumerate(img_files):
        image = cv2.imread(os.path.sep.join([path, file]))
        image.resize(image.size)
        #print(image.size)
        data[i,...]=image
    return data


def resize_image(source_dir, target_dir, width, height):
    img_files = os.listdir(source_dir)
    for file in img_files:
        image = cv2.imread(os.path.sep.join([source_dir, file]))
        print(image.shape)
        image_resized = cv2.resize(image, (width,height), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.sep.join([target_dir, file]), image_resized)
    
    
if __name__ == "__main__":
    #resize_image("../data/orig/benign", "../data/resized/benign", 133, 100)
    #resize_image("../data/orig/malignant", "../data/resized/malignant", 133, 100)    
    data = read_image("../data/resized/benign")
