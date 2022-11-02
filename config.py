import numpy as np
import os
###data path
train_img_txt =r'F:\open_dataset\voc07+12\VOCdevkit\train\train.txt'
train_img_path = '/mnt/data/coco_dataset/train2017'
train_label_path = '/mnt/data/coco_dataset/annotations/instances_train2017.json'
train_num=len(os.listdir(train_img_path))
print(train_num)
# train_num = 5000

test_img_txt = r'F:\open_dataset\voc07+12\VOCdevkit\test\test.txt'
test_img_path = r'F:\open_dataset\voc07+12\VOCdevkit\test\JPEGImages'
test_label_path = r'F:\open_dataset\voc07+12\VOCdevkit\test\Annotations'
test_num = 4952#5011
classes = [
    'back_ground', 'person', 'bicycle', 'car', 'motorcycle',
    'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
    'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli',
    'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet',
    'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush']

# print(len(classes))

##save/restore path
cache_path = './pkl'
val_restore_path =  './checkpoint/model.ckpt-24989'
train_restore_path = '/home/vipuser/workspace/FCOS/pretrained/resnet_v1_50.ckpt'
ckecpoint_file = './checkpoint'


#data aug
gridmask = False
random_crop = True
other_aug = True
multiscale = False#not support now
class_to_ind = 0

weight_decay = 0.0001
momentum_rate = 0.9
gradient_clip_by_norm = 10.0

strides = [8, 16, 32, 64, 128]
limit_range=[[-1,64],[64,128],[128,256],[256,512],[512,999999]]
score_threshold=0.05
nms_iou_threshold=0.3
max_detection_boxes_num = 100


batch_size = 8
LR = 2e-3#0.01/(16/batch_size)
DECAY_STEP = [train_num//batch_size*12, train_num//batch_size*14]
class_weight = 1.
regress_weight = 1.
cnt_weight = 1.
cnt_branch = True#whether use centerness branch

class_num = len(classes)
image_size_h = 512
image_size_w = 640