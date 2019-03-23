#!/usr/bin/env python
# coding: utf-8

# In[10]:


import yolo_modified as yolo
import cv2
import numpy as np
import os


# In[11]:

if "__main__":
    model = yolo.YOLO()


    # In[12]:


    label_path = "model_data/MIR_test.txt"


    # In[13]:


    with open(label_path) as f:
        content = f.readlines()
    ground_truth = {i.split(' ')[0]:np.array([[int(a) for a in aa[0:4]]
                                    for aa in [x.split(',') for x in i.split(' ')[1:]]]) for i in content}


    # In[14]:


    for i in os.listdir('PPMI_/test'):
        img_path = 'PPMI_/test/'+i
        print(img_path)
        frame = cv2.imread(img_path)
        image = yolo.Image.fromarray(frame)
        out_boxes, out_scores, out_classes = model.detect_image_rectang(image)
        image = model.draw_rect(image,out_boxes,out_scores,out_classes)
        image = np.asarray(image)

        for box in ground_truth[img_path]:
            image = cv2.rectangle(image, tuple(box[0:2]), tuple(box[2:4]),color=(0,0,255),thickness=5)
            cv2.imwrite('PPMI_/test_pred/'+i, image)
