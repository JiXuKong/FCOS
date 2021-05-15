import tensorflow as tf
import cv2
import copy
from tensorflow.python import pywrap_tensorflow
import numpy as np
import os, sys
from data.pascal_voc import pascal_voc
import config as cfg 
from model import fcos
from model import fpn_neck
from model import head
from model import loss
from model.timer import Timer


slim = tf.contrib.slim

input_ = tf.placeholder(tf.float32, shape = [1, cfg.image_size_h, cfg.image_size_w, 3])
# get_boxes = tf.placeholder(tf.float32, shape = [cfg.batch_size, 80, 4])
# get_classes = tf.placeholder(tf.float32, shape = [cfg.batch_size, 80])


out = fcos.FCOS(False).forward(input_)
if cfg.cnt_branch:
    pred_class, pred_cnt, pred_reg = out
else:
    pred_class, pred_reg = out
_cls_scores, _cls_classes, _boxes = fcos.DetectHead(cfg.score_threshold, cfg.nms_iou_threshold, cfg.max_detection_boxes_num, cfg.strides).forward(out)
nms_box, nms_score, nms_label = _boxes, _cls_scores, _cls_classes

restore_path = cfg.val_restore_path
g_list = tf.global_variables()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

if restore_path is not None:
    print('Restoring weights from: ' + restore_path)
    restorer = tf.train.Saver(g_list)
    restorer.restore(sess, restore_path)
    
if __name__ == '__main__':
    total_timer = Timer()
    save_fpath = r'E:\FCOS\assets'
    imgnm = './duche2.jpeg'
    img = cv2.imread(imgnm)

    y, x = img.shape[0:2]

    resize_scale_x = 1#x/cfg.image_size
    resize_scale_y = 1#y/cfg.image_size
    

    min_side, max_side    = cfg.image_size_h, cfg.image_size_w
    h, w, _  = img.shape
        
    scale = min_side/h
    if scale*w>max_side:
        scale = max_side/w
    nw, nh  = int(scale * w), int(scale * h)
    print(nh, nw)
    image_resized = cv2.resize(img, (nw, nh))

    paded_w = (cfg.image_size_w - nw)//2
    paded_h = (cfg.image_size_h - nh)//2

    image_paded = np.zeros(shape=[cfg.image_size_h, cfg.image_size_w, 3],dtype=np.uint8)
    image_paded[paded_h:(paded_h+nh), paded_w:(paded_w+nw), :] = image_resized
    img = image_paded
    
    img_orig = copy.deepcopy(img)
    
    img=img[:,:,::-1]
    img=img.astype(np.float32, copy=False)
    mean = np.array([123.68, 116.779, 103.979])
    std = np.array([58.40, 57.12, 57.38])
    mean = mean.reshape((1,1,3))
    std = std.reshape((1,1,3))
    img = img - mean
    img = np.reshape(img, (1, cfg.image_size_h, cfg.image_size_w, 3))
    feed_dict = {
        input_: img
                }
    b, s, l = sess.run([nms_box, nms_score, nms_label], feed_dict = feed_dict)    
    pred_b = b.reshape(-1, 4)
    pred_s = s.reshape(-1,)
    pred_l = l.reshape(-1,)
    for j in range(pred_b.shape[0]):
        if (pred_s[j]>=0.3):
            print(pred_l[j], pred_s[j])
            x1,y1, x2, y2 = pred_b[j][0]*resize_scale_x, pred_b[j][1]*resize_scale_y, pred_b[j][2]*resize_scale_x, pred_b[j][3]*resize_scale_y
            cv2.rectangle(img_orig,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
            cv2.putText(img_orig, str(cfg.classes[pred_l[j]+1]) + str(pred_s[j])[:5],(int(x1),int(y1)),cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),1,1)
    
    img_orig = cv2.resize(img_orig, (640, 512))
    cv2.imshow("x",img_orig)
    cv2.waitKey(0)

