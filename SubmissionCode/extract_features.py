import os
import cv2
import numpy as np

def get_features(image, mask, contour):
    moments = cv2.moments(contour)
    contour_area = cv2.countNonZero(mask)

    contour_centroid = [int(moments['m10'] / moments['m00']),
                        int(moments['m01'] / moments['m00'])]
    contour_perimeter = cv2.arcLength(contour, True)
    
    BI_index = round((contour_perimeter ** 2) / (4 * np.pi * contour_area), 2)

    rect = cv2.fitEllipse(contour)
    (x, y) = rect[0]
    (w, h) = rect[1]
    angle = rect[2]

    if w < h:
        if angle < 90:
            angle -= 90
        else:
            angle += 90
    rows, cols = mask.shape
    rot = cv2.getRotationMatrix2D((x, y), angle, 1)
    cos = np.abs(rot[0, 0])
    sin = np.abs(rot[0, 1])
    W = int((rows * sin) + (cols * cos))
    H = int((rows * cos) + (cols * sin))

    rot[0, 2] += (W / 2) - cols / 2
    rot[1, 2] += (H / 2) - rows / 2

    warp_mask = cv2.warpAffine(mask, rot, (H, W))
    cnts, _ = cv2.findContours(warp_mask, cv2.RETR_TREE,
                                                cv2.CHAIN_APPROX_NONE)
    areas = [cv2.contourArea(c) for c in cnts]
    contour = cnts[np.argmax(areas)]
    xx, yy, W, H = cv2.boundingRect(contour)
    warp_mask = warp_mask[yy:yy + H, xx:xx + W]

    # get asymmetry
    flip_h = cv2.flip(warp_mask, 1)
    flip_v = cv2.flip(warp_mask, 0)

    diff_horizontal = cv2.compare(warp_mask, flip_h,
                                  cv2.CV_8UC1)
    diff_vertical = cv2.compare(warp_mask, flip_v,
                                cv2.CV_8UC1)

    diff_horizontal = cv2.bitwise_not(diff_horizontal)
    diff_vertical = cv2.bitwise_not(diff_vertical)

    h_asym = cv2.countNonZero(diff_horizontal)
    v_asym = cv2.countNonZero(diff_vertical)

    return {'area': int(contour_area), 'perimeter': int(contour_perimeter),
             'BI_index': BI_index,
             'h_diameter': max([W, H]), 'v_diameter': min([W, H]),
             'h_asymmetry': round(float(h_asym) / contour_area, 2),
             'v_asymmetry': round(float(v_asym) / contour_area, 2)}

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
    
def get_contour(img_orig):
    img_gray = cv2.cvtColor(img_orig,cv2.COLOR_BGR2GRAY)
#     cv2.imshow('img_gray',img_gray)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
    _,thresh = cv2.threshold(img_gray,127,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
   
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
    return contour_mask, contour


def segment(input_path, output_path):       
    img_orig = cv2.imread(input_path)
    
    contour_mask, contour = get_contour(img_orig)

    segmented_img = cv2.bitwise_and(
                        img_orig, img_orig,
                        mask=contour_mask)
#     cv2.imshow('segmented_img', segmented_img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
    cv2.imwrite(output_path, segmented_img)
    return segmented_img


def segment_images(source_dir, target_dir):
    img_files = os.listdir(source_dir)
    for file in img_files:
        input_path = os.path.sep.join([source_dir, file])
        output_path = os.path.sep.join([target_dir, file])
        segment(input_path, output_path)

        
def extract_features(source_dir, out_file):
    img_files = os.listdir(source_dir)
    f = open(out_file, 'w')
    
    for file in img_files:
        print(file)
        input_path = os.path.sep.join([source_dir, file])
        img_orig = cv2.imread(input_path)
        contour_mask, contour = get_contour(img_orig)
        feature_dict = get_features(img_orig, contour_mask, contour)
        #print(feature_dict)   
        for key in feature_dict.keys():
            f.write("%s" % (feature_dict[key]))
            if(key!="v_asymmetry"):
                f.write("\t")
        f.write("\n")
    f.close()
        
if __name__ == "__main__":
#     segment_images("../data/resized/benign", "../data/segmented/benign") 
#     segment_images("../data/resized/malignant", "../data/segmented/malignant")
    extract_features("../data/resized/benign","../data/benign_features.txt")
    extract_features("../data/resized/malignant","../data/malignant_features.txt")
