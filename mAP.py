#!/usr/bin/env python
# coding: utf-8

# In[97]:


import yolo_modified
import cv2
import numpy as np
import os


# In[2]:


model = yolo_modified.YOLO()
image_f_path="PPMI_/test"
label_path = "model_data/MIR_test.txt"


# In[76]:


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


# In[105]:


def count_t_positive(boxA, boxB, threshold = 0.5):
# Assume that boxA < boxB
    TP = 0
    for ground_t in boxA:
        min_ind = 0
        ind = 0
        min_dist = float('Inf')
        for out_box in boxB:
            predict_box = np.array([out_box[1],out_box[0],out_box[3],out_box[2]])
            dist = np.linalg.norm(predict_box-ground_t)
            if dist < min_dist:
                min_ind = ind
                min_dist = dist
            ind += 1

        closest_out_box = boxB[min_ind]
        closest_out_box_m = np.array([closest_out_box[1],closest_out_box[0],closest_out_box[3],closest_out_box[2]])
        if bb_intersection_over_union(ground_t,closest_out_box_m)>0.5:
            TP+=1
    return TP


# In[106]:


with open(label_path) as f:
    content = f.readlines()
    ground_truth = {i.split(' ')[0]:np.array([[int(aa[0]),int(aa[1]),int(aa[2]),int(aa[3])]
                                              for aa in [x.split(',') for x in i.split(' ')[1:]]]) for i in content}


# In[ ]:


if "__main__":
    ground_truth_N = 0
    prediction_N = 0

    False_positive = 0
    True_positive = 0
    False_negative = 0

    for image_path in os.listdir(image_f_path):
        img_path = "PPMI_/test/"+image_path
        frame = cv2.imread(img_path)
        image = yolo_modified.Image.fromarray(frame)
        out_boxes, __ , __ = model.detect_image_rectang(image)
        ground_truth_boxes = ground_truth[img_path]

        N_p = len(out_boxes)
        N_gt = len(ground_truth_boxes)

        prediction_N += N_p
        ground_truth_N += N_gt

        if N_p > N_gt:
            False_positive +=  (N_p - N_gt)


        elif N_p < N_gt:
            False_negative +=  (N_gt - N_p)


        if N_p != 0 and N_gt != 0:
            True_positive += count_t_positive(ground_truth_boxes,out_boxes)
    print("Number of Observations: "+ground_truth_N)
    print("Number of Predictions: "+ prediction_N)
    print("Number of False Positives: "+ False_positive)
    print("Number of False Negatives:" + False_negative)
    print("Number of True Positives: "+ True_positive)


# In[ ]:
