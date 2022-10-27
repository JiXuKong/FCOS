from data.augment.aug_python import _crop, random_color_distort
# from data.augment import *
from data.augment.data_aug import Rotate, Sequence, RandomRotate, RandomHorizontalFlip, RandomScale, RandomTranslate, RandomShear
import config as cfg

import xml.etree.ElementTree as ET
import numpy as np
import random
import pickle
import os
import cv2

# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, division
import sys, os

from data.cocoapi.PythonAPI.pycocotools.coco import COCO
from data import coco_dict


class pascal_voc(object):
    def __init__(self, phase, flipped, img_path, label_path, , is_training, ssl = False):
        self.is_training = is_training
        self.img_path = img_path
        self.label_path = label_path
        self.cache_path = cfg.cache_path
        self.img_size = cfg.image_size
        self.batch_size = cfg.batch_size
        self.classes = cfg.classes
        self.class_to_ind = dict(zip(self.classes, range(len(self.classes))))
        self.flipped = flipped
        self.gt_labels = None
        self.phase = phase
        self.epoch = 1
        self.corsor = 0
        self.cnt = 0
        self.ssl = ssl
        self.data_augmentation()
        
    def load_annotation(self,annotation):
        gtbox_and_label_list = []
        for ann in annotation:
            box = ann['bbox']
            box = [box[0], box[1], box[0]+box[2], box[1]+box[3]]  # [xmin, ymin, xmax, ymax]
            cat_id = ann['category_id']
            cat_name = coco_dict.originID_classes[cat_id] #ID_NAME_DICT[cat_id]
            label = coco_dict.NAME_LABEL_MAP[cat_name]
            gtbox_and_label_list.append([label] + box)
        gtbox_and_label_list = np.array(gtbox_and_label_list, dtype=np.int32)
        return gtbox_and_label_list, len(gtbox_and_label_list)
        
    def read_image(self,imgnm, bboxes, img_size, flipped = False):
        img = cv2.imread(imgnm)
        y, x = img.shape[0:2]
#         print('img_size:', img_size)
        img = cv2.resize(img,(img_size[0],img_size[1]))
        resize_scale_x = img_size[1]/x
        resize_scale_y = img_size[0]/y
        x1 = bboxes[:, 0]
        y1 = bboxes[:, 1]
        x2 = bboxes[:, 2]
        y2 = bboxes[:, 3]
        x1 *= resize_scale_x
        y1 *= resize_scale_y
        x2 *= resize_scale_x
        y2 *= resize_scale_y
        bboxes[:, 0] = x1
        bboxes[:, 1] = y1
        bboxes[:, 2] = x2
        bboxes[:, 3] = y2
        #then bgrtorgb
        img=img[:,:,::-1]
        img=img.astype(np.float32, copy=False)
        
        #######################random crop#####################
        if cfg.random_crop and self.is_training:
            p = random.random()
            if p>0.5:# and self.ssl:
                box_label = bboxes[:,:4]
                class_label = bboxes[:, 4]

                image_t, boxes_t, labels_t = _crop(img, box_label, class_label)
                t_h, t_w = image_t.shape[:2]
                t_x1 = boxes_t[:, 0]*img_size[1]/t_w
                t_y1 = boxes_t[:, 1]*img_size[0]/t_h
                t_x2 = boxes_t[:, 2]*img_size[1]/t_w
                t_y2 = boxes_t[:, 3]*img_size[0]/t_h
                boxes_t = np.vstack((t_x1, t_y1, t_x2, t_y2)).transpose()
                bboxes = np.append(boxes_t, labels_t.reshape(-1, 1), axis = 1)

                img = image_t
                img = cv2.resize(img,(img_size[0],img_size[1]))
        #######################random crop##################### 
    
        ######################augment stratage 2###############
        if cfg.other_aug and self.is_training:# and self.ssl:
            # p = random.random()
            # if p>0.5:
            #     img_, bboxes_ = Rotate(90)(img, bboxes)
            #     if bboxes_.shape[0] != 0:
            #         bboxes = bboxes_
            #         img = img_
            p = random.random()
            if p>0.8:
                seq = Sequence([RandomRotate(1), RandomHorizontalFlip(), RandomScale(), RandomTranslate(), RandomShear()])
                img_, bboxes_ = seq(img, bboxes)
                if bboxes_.shape[0] != 0:
                    bboxes = bboxes_
                    img = img_
            p = random.random()
            if p>0.5:
                img = random_color_distort(img)
        ######################augment stratage 2###############    

        
        mean = np.array([123.68, 116.779, 103.979])
#         mean = np.array([68.47, 68.47, 68.47])
        mean = mean.reshape(1,1,3)
        img = img - mean
        return img, bboxes
    
    def load_labels(self):
        gt_labels = []
        d = 0
        
        print ("load coco .... it will cost about 17s..")
        coco = COCO(self.label_path)
        imgId_list = coco.getImgIds()
        imgId_list = np.array(imgId_list)
        # np.random.shuffle(imgId_list)
        self.imgId_list = imgId_list
        

        total_imgs = len(imgId_list)

        for step in range(total_imgs):
            imgid = imgId_list[step]
            imgname = coco.loadImgs(ids=[imgid])[0]['file_name']
            imgname = os.path.join(self.img_path, imgname)

            annotation = coco.imgToAnns[imgid]
            label,num = self.load_annotation(annotation)

            gt_labels.append({
                     'label' : label,
                     'img_dir' : imgname,
                     'flipped' : False
                    })
        return gt_labels
            
    def data_augmentation(self):
        gt_labels = self.load_labels()
        np.random.shuffle(gt_labels)
        self.gt_labels = gt_labels
    

        
        
    def get(self,):

        batch_imnm = []
        num_boxes = []
        images = []
        labels = np.zeros((self.batch_size,80,5))
        count = 0
        if cfg.multiscale and self.is_training:
            random.seed(self.cnt // 4000)#有参数，每次生成的随机数相同
            random_img_size = [[x * 32, x * 32] for x in range(15, 25)]
            img_size = random.sample(random_img_size, 1)[0]
            img_size = [img_size, img_size]
        else:    
            img_size = [self.img_size, self.img_size]
        while count < self.batch_size:
            imnm = self.gt_labels[self.corsor]['img_dir']

            label = self.gt_labels[self.corsor]['label']

            label = np.append(label[:,1:],label[:,0].reshape(-1, 1), axis = 1)

            image, label = self.read_image(imnm, label, img_size)
            label = np.append(label[:,4].reshape(-1, 1), label[:,:4], axis = 1)
            if np.where(label<0)[0].shape[0]>0:
                continue
            labels[count,:label.shape[0],:] = label
            num_boxes.append(label.shape[0])
            images.append(image)

            batch_imnm.append(imnm)
            count += 1
            self.corsor += 1
            if self.corsor >= len(self.gt_labels):
                np.random.shuffle(self.gt_labels)
                self.corsor = 0
                self.epoch += 1
                
      
        return np.asarray(images), labels, batch_imnm, num_boxes, np.array(img_size)
        # return np.asarray(images), labels, batch_imnm, num_boxes, [np.array(img_size)

#use example:        
p = pascal_voc(phase='train', flipped=True, img_path = cfg.train_img_path, label_path=cfg.train_label_path, img_txt=cfg.train_img_txt, is_training=True)
_ = p.load_labels()
