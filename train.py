import tensorflow as tf
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
from model.tool import show_box_in_tensor

slim = tf.contrib.slim

data = pascal_voc('train', False, cfg.train_img_path, cfg.train_label_path, cfg.train_img_txt, True)


input_ = tf.placeholder(tf.float32, shape = [cfg.batch_size, cfg.image_size_h, cfg.image_size_w, 3])
get_boxes = tf.placeholder(tf.float32, shape = [cfg.batch_size, 80, 4])
get_classes = tf.placeholder(tf.float32, shape = [cfg.batch_size, 80])


out = fcos.FCOS(True).forward(input_)
targets = loss.GenTargets(cfg.strides, cfg.limit_range).forward((out, get_boxes, get_classes))
if cfg.cnt_branch:
    pred_class, pred_cnt, pred_reg = out
    inputs = out, targets
    cls_loss, cnt_loss, reg_loss = loss.LOSS().forward(inputs)
else:
    pred_class, pred_reg = out
    inputs = out, targets
    cls_loss, reg_loss = loss.LOSS().forward(inputs)
# total_loss = cls_loss*cfg.class_weight + cnt_loss*cfg.cnt_weight + reg_loss*cfg.regress_weight
with tf.name_scope('weight_decay'):
    slim_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    regularization_loss = tf.losses.get_regularization_loss()
    
with tf.name_scope('class_loss'):    
    tf.losses.add_loss(cls_loss*cfg.class_weight)
if cfg.cnt_branch:
    with tf.name_scope('cnt_loss'):      
        tf.losses.add_loss(cnt_loss*cfg.cnt_weight)
with tf.name_scope('giou_loss'):     
    tf.losses.add_loss(reg_loss*cfg.regress_weight)
with tf.name_scope('total_loss'):     
    total_loss = tf.losses.get_total_loss(add_regularization_losses=True)


with tf.name_scope('detection'): 
    cls_1 = []
    cnt_1 = []
    reg_1 = []
    for level in range(len(out[0])):
        cls_1.append(tf.expand_dims(out[0][level][0], axis = 0))
        if cfg.cnt_branch:
            cnt_1.append(tf.expand_dims(out[1][level][0], axis = 0))
            reg_1.append(tf.expand_dims(out[2][level][0], axis = 0))
            temp = tf.stop_gradient(tf.nn.sigmoid(tf.expand_dims(out[1][level][0], axis = 0)))
            tf.summary.image('level_'+str(level)+'cntMAP', temp)
        else:
            reg_1.append(tf.expand_dims(out[1][level][0], axis = 0))
        
    if cfg.cnt_branch:
        _cls_scores, _cls_classes, _boxes = fcos.DetectHead(cfg.score_threshold, cfg.nms_iou_threshold, cfg.max_detection_boxes_num, cfg.strides).forward([cls_1, cnt_1, reg_1])
    else:
        _cls_scores, _cls_classes, _boxes = fcos.DetectHead(cfg.score_threshold, cfg.nms_iou_threshold, cfg.max_detection_boxes_num, cfg.strides).forward([cls_1, reg_1])
    
    nms_box, nms_score, nms_label = tf.reshape(_boxes[0], [-1, 4]), tf.reshape(_cls_scores[0], [-1,]), tf.reshape(_cls_classes[0], [-1,])
    detection_in_img = show_box_in_tensor.draw_boxes_with_categories_and_scores(img_batch=tf.expand_dims(input_[0], 0),
                                                             boxes=nms_box,
                                                             labels=nms_label+1,
                                                             scores=nms_score)
    gt_in_img = show_box_in_tensor.draw_boxes_with_categories_and_scores(img_batch=tf.expand_dims(input_[0], 0),
                                                             boxes=get_boxes[0],
                                                             labels=get_classes[0],
                                                             scores=get_classes[0])
    tf.summary.image('detection', detection_in_img)
    tf.summary.image('GT', gt_in_img)

global_step = slim.get_or_create_global_step()
with tf.variable_scope('learning_rate'):
    lr = tf.train.piecewise_constant(global_step,
                                     boundaries=[np.int64(cfg.DECAY_STEP[0]), np.int64(cfg.DECAY_STEP[1])],
                                     values=[cfg.LR, cfg.LR / 10., cfg.LR / 100.])
#     LR = 1e-5
#     lr = tf.train.piecewise_constant(global_step,
#                                      boundaries=[np.int64(1.01e5), np.int64(1.02e5), np.int64(1.07e5), np.int64(1.12e5)],
#                                      values=[LR, LR * 10., LR * 100., LR * 10., LR])

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer = tf.train.MomentumOptimizer(lr, cfg.momentum_rate, use_nesterov=False)
#     optimizer = tf.train.AdamOptimizer(lr)
    gradient = optimizer.compute_gradients(total_loss)
    with tf.name_scope('clip_gradients_YJR'):
        gradient = slim.learning.clip_gradient_norms(gradient,cfg.gradient_clip_by_norm)

    with tf.name_scope('apply_gradients'):
        train_op = optimizer.apply_gradients(grads_and_vars=gradient,global_step=global_step)

g_list = tf.global_variables()
# for g in g_list:
#     print(g.name)AdamMomentum
save_list = [g for g in g_list if ('Momentum' not in g.name)and('ExponentialMovingAverage' not in g.name)]
saver = tf.train.Saver(var_list=save_list, max_to_keep=30)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
summary_op = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter(cfg.ckecpoint_file, sess.graph)


def get_variables_in_checkpoint_file(file_name):
    reader = pywrap_tensorflow.NewCheckpointReader(file_name)
    var_to_shape_map = reader.get_variable_to_shape_map()
    return var_to_shape_map

def initialize(pretrained_model, variable_to_restore):
    var_keep_dic = get_variables_in_checkpoint_file(pretrained_model)
    # Get the variables to restore, ignoring the variables to fix
    variables_to_restore = get_variables_to_restore(variable_to_restore, var_keep_dic)
    restorer = tf.train.Saver(variables_to_restore)
    return restorer

def get_variables_to_restore(variables, var_keep_dic):
    variables_to_restore = []

    for v in variables:
        if (v.name == 'global_step:0'):
            continue;

#         if(v.name.split('/')[1] != 'ClassPredictor')\
#         and(v.name.split('/')[1] != 'BoxPredictor')\
#         and(v.name.split(':')[0])in var_keep_dic:
#         if 'WeightSharedConvolutionalBoxPredictor' not in v.name\
#          and 'FeatureExtractor' not in v.name\
        
        if (v.name.split(':')[0])in var_keep_dic:

            print('Variables restored: %s' % v.name)
            variables_to_restore.append(v)
        
    return variables_to_restore

if cfg.train_restore_path is not None:
    print('Restoring weights from: ' + cfg.train_restore_path)
    restorer = initialize(cfg.train_restore_path, g_list)
#     restorer = tf.train.Saver(save_list)
    restorer.restore(sess, cfg.train_restore_path)
    

def train():
    total_timer = Timer()
    train_timer = Timer()
    load_timer = Timer()
    max_epoch = 30
    epoch_step = int(cfg.train_num//cfg.batch_size)
    t = 1
    for epoch in range(1, max_epoch + 1):
        print('-'*25, 'epoch', epoch,'/',str(max_epoch), '-'*25)


        t_loss = 0
        ll_loss = 0
        r_loss = 0
        c_loss = 0
        
        
       
        for step in range(1, epoch_step + 1):
     
            t = t + 1
            total_timer.tic()
            load_timer.tic()
 
            images, labels, imnm, num_boxes, imsize = data.get()
            
#             load_timer.toc()
            feed_dict = {input_: images,
                         get_boxes: labels[..., 1:5][:, ::-1, :],
                         get_classes: labels[..., 0].reshape((cfg.batch_size, -1))[:, ::-1]
                        }
            if cfg.cnt_branch:
                _, g_step_, tt_loss, cl_loss, cn_loss, re_loss, lr_ = sess.run(
                    [train_op,
                     global_step,
                     total_loss,
                     cls_loss, 
                     cnt_loss, 
                     reg_loss,
                     lr], feed_dict = feed_dict)
            else:
                _, g_step_, tt_loss, cl_loss, re_loss, lr_ = sess.run(
                    [train_op,
                     global_step,
                     total_loss,
                     cls_loss, 
                     reg_loss,
                     lr], feed_dict = feed_dict)
            
            
            total_timer.toc()
            if g_step_%50 ==0:
                sys.stdout.write('\r>> ' + 'iters '+str(g_step_)+str('/')+str(epoch_step*max_epoch)+' loss '+str(tt_loss) + ' ')
                sys.stdout.flush()
                summary_str = sess.run(summary_op, feed_dict = feed_dict)
                
                train_total_summary = tf.Summary(value=[
                    tf.Summary.Value(tag="config/learning rate", simple_value=lr_),
                    tf.Summary.Value(tag="train/classification/focal_loss", simple_value=cfg.class_weight*cl_loss),
                    tf.Summary.Value(tag="train/classification/cnt_loss", simple_value=cfg.cnt_weight*cn_loss),
#                     tf.Summary.Value(tag="train/p_nm", simple_value=p_nm_),
                    tf.Summary.Value(tag="train/regress_loss", simple_value=cfg.regress_weight*re_loss),
#                     tf.Summary.Value(tag="train/clone_loss", simple_value=cfg.class_weight*cl_loss + cfg.regress_weight*re_loss + cfg.cnt_weight*cn_loss),
#                     tf.Summary.Value(tag="train/l2_loss", simple_value=l2_loss),
                    tf.Summary.Value(tag="train/total_loss", simple_value=tt_loss)
                    ])
                print('curent speed: ', total_timer.diff, 'remain time: ', total_timer.remain(g_step_, epoch_step*max_epoch))
                summary_writer.add_summary(summary_str, g_step_)
                summary_writer.add_summary(train_total_summary, g_step_)
            if g_step_%10000 == 0:
                print('saving checkpoint')
                saver.save(sess, cfg.ckecpoint_file + '/model.ckpt', g_step_)

        total_timer.toc()
        sys.stdout.write('\n')
        print('>> mean loss', t_loss)
        print('curent speed: ', total_timer.average_time, 'remain time: ', total_timer.remain(g_step_, epoch_step*max_epoch))
        
    print('saving checkpoint')
    saver.save(sess, cfg.ckecpoint_file + '/model.ckpt', g_step_)
    
    
train()    
    
    
    
    
    