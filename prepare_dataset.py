#%%
import numpy as np 
import pandas as pd

import random
from PIL import Image
from matplotlib import pyplot as plt
import xml.etree.ElementTree as ET

import torch 
import torchvision

import cv2
import os

# %%
import os
def fn(fol):       # 1.Get file names from directory
    file_list=os.listdir(fol)
    return file_list

# %%
size=300 #70
test_or_train="train"
if (test_or_train=="train"):
    img_base_path = "data"
    dir_out="output/"
    annotations_folder="annotations"
else:
    img_base_path = "original_test/left"
    dir_out="output_test/"
    annotations_folder="test_annotations/left-labeled"
obj_type_list=['book', 'box', 'cup']
box_it=0
book_it=0
cup_it=0

fls=fn(annotations_folder)
for fl in fls:
    annotation_path=annotations_folder+'/'+fl
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    objects = root.findall('object')

    image_name=root.find('filename').text
    img_org = Image.open(os.path.join(img_base_path, image_name))
    it=0
    for obj in objects:        
        obj_type=obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        bbox = (xmin, ymin, xmax, ymax)
        # print("Bounding Box: ", bbox)
        print(obj_type)
        img = img_org.crop(bbox)
        if (test_or_train=="train"):
            new_img = cv2.resize(np.array(img), dsize=(size, size), interpolation=cv2.INTER_CUBIC)
        else:
            new_img=np.array(img)
        im_bgr = cv2.cvtColor(np.array(new_img), cv2.COLOR_RGB2BGR)
        
        file_name=os.path.splitext(image_name)[0]+str(it)+".jpg"  

        if(obj_type==obj_type_list[0]):
            cv2.imwrite(dir_out+obj_type+"/"+file_name, im_bgr)
        if(obj_type==obj_type_list[1]):
            cv2.imwrite(dir_out+obj_type+"/"+file_name, im_bgr)
        if(obj_type==obj_type_list[2]):   
            cv2.imwrite(dir_out+obj_type+"/"+file_name, im_bgr)
        it+=1
        # fig = plt.figure(figsize=(8, 12))
        # plt.imshow(im_bgr)

# %%
