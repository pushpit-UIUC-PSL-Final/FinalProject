'''
Created on Dec 14, 2019

@author: HARISH C
'''
import os
import cv2
import numpy as np



def get_largest_contour(gray_image):
    mask_contours, _ = cv2.findContours(gray_image, cv2.RETR_EXTERNAL,
                         cv2.CHAIN_APPROX_NONE)
    cnt = len(mask_contours)
    if cnt > 0:
        area = np.zeros(cnt)
        for i in np.arange(cnt):
            area[i] = cv2.contourArea(mask_contours[i])
        max_area_pos = np.argpartition(area, -1)[-1:][0]
        return mask_contours[max_area_pos]
    else:
        return None

def segment(input_path, output_path):       
    img_orig = cv2.imread(input_path)
    
    img_gray = cv2.cvtColor(img_orig,cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(img_gray,127,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
#     cv2.imshow('img_gray',img_gray)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
   
    contour_mask = np.full( img_gray.shape, 255,np.uint8 )
    cv2.fillPoly(contour_mask, pts =contours, color=(0,0,0))
#     cv2.imshow('contour_mask',contour_mask)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     
    contour = get_largest_contour(contour_mask)
    mask = np.zeros(img_gray.shape, np.uint8)
    contour_img = cv2.drawContours(mask, contour, -1, (255,255,255), 1)
#     cv2.imshow('contour_img',contour_img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

    segmented_img = cv2.bitwise_and(
                        img_orig, img_orig,
                        mask=contour_mask)
#     cv2.imshow('segmented_img', segmented_img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
    cv2.imwrite(output_path, segmented_img)
    print(segmented_img.shape)
    return segmented_img

def segment_images(source_dir, target_dir):
    img_files = os.listdir(source_dir)
    for file in img_files:
        input_path = os.path.sep.join([source_dir, file])
        output_path = os.path.sep.join([target_dir, file])
        segment(input_path,output_path)
        
if __name__ == "__main__":
    segment_images("../data/resized/benign", "../data/segmented/benign") 
    segment_images("../data/resized/malignant", "../data/segmented/malignant") 
