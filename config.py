import numpy as np

###data path
train_img_txt =r'F:\open_dataset\voc07+12\VOCdevkit\train\train.txt'
train_img_path = r'F:\open_dataset\voc07+12\VOCdevkit\train\JPEGImages'
train_label_path = r'F:\open_dataset\voc07+12\VOCdevkit\train\Annotations'
train_num = 5000

test_img_txt = r'F:\open_dataset\voc07+12\VOCdevkit\test\test.txt'
test_img_path = r'F:\open_dataset\voc07+12\VOCdevkit\test\JPEGImages'
test_label_path = r'F:\open_dataset\voc07+12\VOCdevkit\test\Annotations'
test_num = 4952#5011
classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor']

##save/restore path
cache_path = './pkl'
val_restore_path =  './checkpoint/model.ckpt-24989'
train_restore_path = r'pretrained/resnet_v1_50.ckpt'
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


batch_size = 1
LR = 2e-3#0.01/(16/batch_size)
DECAY_STEP = [train_num//batch_size*20, train_num//batch_size*27]
class_weight = 1.
regress_weight = 1.
cnt_weight = 1.
cnt_branch = True#whether use centerness branch

class_num = len(classes)
image_size_h = int(512)
image_size_w = int(640)