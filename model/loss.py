import tensorflow as tf
import numpy as np
import config as cfg
import math

'''
py_:python接口
coords_fmap2orig:根据特征和步长映射回原图的coords
GenTargets:目标标签

'''


def py_(batch_size, m, areas, mask_pos, ltrb_off, classes):
    '''
    areas: [batch_size,h*w, m]
    mask_pos:
    ltrb_off:[batch_size,h*w, m, 4]
    classes:
    '''
    areas[~mask_pos]=99999999
    areas_min_ind = np.argmin(areas, axis = -1)#[batch_size,h*w]
#     print(np.where(areas<=0))
#     print(np.argmin(areas, axis = -1))
    ltrb_off = np.reshape(ltrb_off, (-1, m, 4))#[batch_size*h*w, m, 4]
    areas_min_ind = np.reshape(areas_min_ind, (-1,))#[batch_size*h*w]
#     reg_targets=ltrb_off[torch.zeros_like(areas,dtype=torch.uint8).scatter_(-1,areas_min_ind.unsqueeze(dim=-1),1)]#[batch_size*h*w,4]
    reg_targets = ltrb_off[np.arange(areas_min_ind.shape[0]), areas_min_ind]
    reg_targets = np.reshape(reg_targets, (batch_size, -1, 4))#[batch_size,h*w,4]
#     print('reg_targets: ', reg_targets.shape)
    
    classes = np.zeros_like(areas) + np.reshape(classes, (batch_size, 1, m))#[b, 1, m] + [b, h*w, m] = [b, h*w, m]
    classes = np.reshape(classes, (-1, m, 1))
    cls_targets = classes[np.arange(areas_min_ind.shape[0]), areas_min_ind]
    cls_targets = np.reshape(cls_targets, (batch_size, -1, 1))
    
    left_right_min = np.minimum(reg_targets[..., 0], reg_targets[..., 2])#[batch_size,h*w]
    left_right_max = np.maximum(reg_targets[..., 0], reg_targets[..., 2])
    top_bottom_min = np.minimum(reg_targets[..., 1], reg_targets[..., 3])
    top_bottom_max = np.maximum(reg_targets[..., 1], reg_targets[..., 3])
#     print(np.where((left_right_min*top_bottom_min)/(left_right_max*top_bottom_max+1e-10)<=0))
    cnt_targets = np.sqrt((left_right_min*top_bottom_min)/(left_right_max*top_bottom_max+1e-10))#[batch_size,h*w,1]
    cnt_targets = np.expand_dims(cnt_targets, axis = -1)
    
    mask_pos_2 = np.sum(mask_pos.astype(np.int64), axis = -1)#[b, h*w]
    mask_pos_2 = mask_pos_2 >= 1
#     print(mask_pos_2.shape, cls_targets.shape)
    
    cls_targets[~mask_pos_2]=0 #[batch_size,h*w,1]
    cnt_targets[~mask_pos_2]=-1#[batch_size,h*w,1]
    reg_targets[~mask_pos_2]=-1#[batch_size,h*w,4]
    cls_targets = np.asarray(cls_targets, dtype=np.float32)
    return cls_targets, cnt_targets, reg_targets

def coords_fmap2orig(feature,stride):
    '''
    transfor one fmap coords to orig coords
    Args
    featurn [batch_size,h,w,c]
    stride int
    Returns 
    coords [n,2]
    '''
#     h, w = tf.shape(feature)[1:3]
    h, w = feature.get_shape().as_list()[1:3]
#     print(h,w)
    shift_x = tf.range(w, dtype=tf.float32)*stride
    shift_y = tf.range(h, dtype=tf.float32)*stride
    
    shift_x, shift_y = tf.meshgrid(shift_x, shift_y)
    shift_x = tf.reshape(shift_x, [-1])
    shift_y = tf.reshape(shift_y, [-1])
    
    coords = tf.stack([shift_x, shift_y], axis = -1) + stride//2
    return coords

class GenTargets(object):
    def __init__(self, strides, limit_range, debug=False):
        self.strides=strides
        self.limit_range=limit_range
        self.debug = debug
        assert len(strides)==len(limit_range)

    def forward(self,inputs):
        '''
        inputs  
        [0]list [cls_logits,cnt_logits,reg_preds]  
        cls_logits  list contains five [batch_size,class_num,h,w]  
        cnt_logits  list contains five [batch_size,1,h,w]  
        reg_preds   list contains five [batch_size,4,h,w]  
        [1]gt_boxes [batch_size,m,4]  FloatTensor  
        [2]classes [batch_size,m]  LongTensor
        Returns
        cls_targets:[batch_size,sum(_h*_w),1]
        cnt_targets:[batch_size,sum(_h*_w),1]
        reg_targets:[batch_size,sum(_h*_w),4]
        '''
        if cfg.cnt_branch:
            cls_logits,cnt_logits,reg_preds=inputs[0]
        else:
            cls_logits, reg_preds=inputs[0]
        gt_boxes=inputs[1]
        classes=inputs[2]
        cls_targets_all_level=[]
        cnt_targets_all_level=[]
        reg_targets_all_level=[]
        assert len(self.strides)==len(cls_logits)
        for level in range(len(cls_logits)):
            level_out=cls_logits[level]
            level_targets=self._gen_level_targets(level_out,gt_boxes,classes,self.strides[level],
                                                    self.limit_range[level])
            cls_targets_all_level.append(level_targets[0])
            cnt_targets_all_level.append(level_targets[1])
            reg_targets_all_level.append(level_targets[2])
        if self.debug:
            return cls_targets_all_level, cnt_targets_all_level, reg_targets_all_level
        else:
            cls_targets_all_level = tf.concat(cls_targets_all_level, axis = 1)
            cnt_targets_all_level = tf.concat(cnt_targets_all_level, axis = 1) 
            reg_targets_all_level = tf.concat(reg_targets_all_level, axis = 1)
    #         print('preds, targets: ', cls_targets_all_level.get_shape().as_list())
            return cls_targets_all_level, cnt_targets_all_level, reg_targets_all_level
        
        
        
    def _gen_level_targets(self,out,gt_boxes,classes,stride,limit_range,sample_radiu_ratio=1.5):
        
        '''
        Args  
        out list contains [[batch_size,h,w,class_num],[batch_size,h,w,1],[batch_size,h,w,4]]  
        gt_boxes [batch_size,m,4]  
        classes [batch_size,m]  
        stride int  
        limit_range list [min,max]  
        Returns  
        cls_targets,cnt_targets,reg_targets
        '''
        cls_logits = out
        batch_size = cls_logits.get_shape().as_list()[0]
        class_num = cls_logits.get_shape().as_list()[-1]
        m = gt_boxes.get_shape().as_list()[1]
        
        coords = coords_fmap2orig(cls_logits, stride)#[h*w,2]
        cls_logits = tf.reshape(cls_logits, [batch_size,-1,class_num])#[batch_size,h*w,class_num]  
#         cnt_logits = tf.reshape(cnt_logits, [batch_size,-1, 1])
#         reg_preds = tf.reshape(reg_preds, [batch_size,-1,4])
#         h_mul_w = cls_logits.shape[1]
        
        x = coords[:, 0]
        y = coords[:, 1]
#         print('  d',y.get_shape().as_list(), x.get_shape().as_list())
        
        x = tf.reshape(x, [1, -1, 1])
        y = tf.reshape(y, [1, -1, 1])
        
        l_off = x - tf.expand_dims(gt_boxes[..., 0], axis = 1)
        t_off = y - tf.expand_dims(gt_boxes[...,1], axis = 1)
        r_off = tf.expand_dims(gt_boxes[..., 2], axis = 1) - x
        b_off = tf.expand_dims(gt_boxes[..., 3], axis = 1) - y
        
        ###############
#         l_off = x[None,:,None] - gt_boxes[...,0][:,None,:]#[1,h*w,1]-[batch_size,1,m]-->[batch_size,h*w,m]
#         t_off = y[None,:,None] - gt_boxes[...,1][:,None,:]
#         r_off = gt_boxes[...,2][:,None,:] - x[None,:,None]
#         b_off = gt_boxes[...,3][:,None,:] - y[None,:,None]
        
        
        
        
        ltrb_off = tf.stack([l_off, t_off, r_off, b_off], axis=-1)#[batch_size,h*w,m,4]
        
        areas = (ltrb_off[..., 0] + ltrb_off[..., 2])*(ltrb_off[..., 1] + ltrb_off[..., 3])#[batch_size,h*w,m]
        
        off_min = tf.reduce_min(ltrb_off, axis = -1, keep_dims = False)#[batch_size,h*w,m]
        off_max =tf.reduce_max(ltrb_off, axis = -1, keep_dims = False)#[batch_size,h*w,m]

        mask_in_gtboxes = off_min>0
        mask_in_level = (off_max > limit_range[0]) & (off_max <= limit_range[1])
        
        radiu = stride*sample_radiu_ratio
        gt_center_x = (gt_boxes[..., 0] + gt_boxes[..., 2])/2
        gt_center_y = (gt_boxes[..., 1] + gt_boxes[..., 3])/2
        
        c_l_off = x - tf.expand_dims(gt_center_x, axis = 1)#[1,h*w,1]-[batch_size,1,m]-->[batch_size,h*w,m]
        c_t_off = y - tf.expand_dims(gt_center_y, axis = 1)
        c_r_off = tf.expand_dims(gt_center_x, axis = 1) - x
        c_b_off = tf.expand_dims(gt_center_y, axis = 1) - y
        
#         c_l_off = x[None, :, None] - gt_center_x[:, None, :]#[1,h*w,1]-[batch_size,1,m]-->[batch_size,h*w,m]
#         c_t_off = y[None, :, None] - gt_center_y[:, None, :]
#         c_r_off = gt_center_x[:, None, :] - x[None, :, None]
#         c_b_off = gt_center_y[:, None, :] - y[None, :, None]
        
        
        
        
        c_ltrb_off = tf.stack([c_l_off, c_t_off, c_r_off, c_b_off], axis = -1)#[batch_size,h*w,m,4]
        c_off_max = tf.reduce_max(c_ltrb_off, axis = -1, keep_dims = False)
        mask_center = c_off_max < radiu
        
        mask_pos = mask_in_gtboxes&mask_in_level&mask_center#[batch_size,h*w, m]
        
        cls_targets, cnt_targets, reg_targets = tf.py_func(py_,
                               [batch_size, m, areas, mask_pos, ltrb_off, classes],
                               [tf.float32, tf.float32, tf.float32])
        return cls_targets,cnt_targets,reg_targets

def focal_loss(predictions, targets, weights, gamma=2.0, alpha=0.25):
    """
    Arguments:
        predictions: a float tensor with shape [batch_size, -1, num_classes],
            representing the predicted logits for each class.
        targets: a float tensor with shape [batch_size, -1, num_classes],
            representing one-hot encoded classification targets.
        weights: a float tensor with shape [batch_size, -1].
        gamma, alpha: float numbers.
    Returns:
        a float tensor with shape [batch_size, num_anchors].
    """
    positive_label_mask = tf.equal(targets, 1.0)

#     delta = 0.01
#     targets = (1 - delta) * targets + + delta * 1. / 2
    negative_log_p_t = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets, logits=predictions)
#     negative_log_p_t = tf.losses.sigmoid_cross_entropy(multi_class_labels=targets, logits = predictions, label_smoothing=0)
    
    
    probabilities = tf.sigmoid(predictions)
    p_t = tf.where(positive_label_mask, probabilities, 1.0 - probabilities)
    # they all have shape [batch_size, num_anchors, num_classes]

    modulating_factor = tf.pow(1.0 - p_t, gamma)
    weighted_loss = tf.where(
        positive_label_mask,
        alpha * negative_log_p_t,
        (1.0 - alpha) * negative_log_p_t
    )
    focal_loss = modulating_factor * weighted_loss
    # they all have shape [batch_size, num_anchors, num_classes]
    weights = tf.cast(weights, tf.float32)
    return weights * tf.reduce_sum(focal_loss, axis=2)
#     return weights * tf.reduce_sum(focal_loss, axis=2)

def focal_loss_from_logits(preds,targets,gamma=2.0,alpha=0.25):
    '''
    Args:
    preds: [n,class_num] 
    targets: [n,class_num]
    '''
    preds = tf.nn.sigmoid(preds)
    pt = preds*targets + (1.0 - preds)*(1.0 - targets)
    w = alpha*targets + (1.0 - alpha)*(1.0 - targets)
    loss =-w*tf.pow((1.0 - pt), gamma)*tf.log(pt)
    return tf.reduce_sum(loss, axis=[0,1])



def compute_cnt_loss(preds, targets, mask):
    
    """
    
    Arguments:
        predictions: a float tensor with shape [batch_size, -1, 1],
            representing the predicted logits for each class.
        targets: a float tensor with shape [batch_size, -1, 1],
            representing one-hot encoded classification targets.
        weights: a float tensor with shape [batch_size, -1].
        gamma, alpha: float numbers.
    Returns:
        a float tensor with shape [batch_size, num_anchors].
    """
#     print('targets: ', targets)
    b = preds[0].get_shape().as_list()[0]
    c = preds[0].get_shape().as_list()[-1]

    preds_reshape = []
    mask = tf.expand_dims(mask, axis = -1)#[batch_size,sum(_h*_w),1]
    # mask=targets>-1#[batch_size,sum(_h*_w),1]
    num_pos = tf.reduce_sum(mask, axis = [1,2])#[batch_size,]
#     num_pos = tf.cast(tf.clip_by_value(num_pos, int(1), int(1e4)), tf.float32)
    num_pos = tf.cast(tf.maximum(num_pos, int(1)), tf.float32)                  
   
    preds_ = []
    for i in range(len(preds)):
        preds_.append(tf.reshape(preds[i], [b, -1, c]))
    preds = preds_
    preds = tf.concat(preds, axis = 1)
    
    loss=[]
    for batch_index in range(b):
        preds_i = preds[batch_index]
#         mask_i = mask[batch_index]
        mask_i = tf.squeeze(mask[batch_index], axis = -1)#[-1,]
        mask_i = tf.where(mask_i > 0)
        mask_i = tf.squeeze(mask_i, axis = -1)
        pred_pos_i = tf.gather(preds_i, mask_i)#[num_pos_b,]
        targets_i = targets[batch_index]
        target_pos_i=tf.gather(targets_i, mask_i)#[num_pos_b,]
        loss_i = tf.nn.sigmoid_cross_entropy_with_logits(labels = target_pos_i, logits = pred_pos_i)
        loss_i = tf.reduce_sum(loss_i, axis = 0)/tf.stop_gradient(num_pos[batch_index])
        loss.append(loss_i)
    return loss#[batch_size,]
        

# def giou_loss(preds, targets, mask):
    
#      '''
#     # Copyright (C) 2019 * Ltd. All rights reserved.
#     # author : SangHyeon Jo <josanghyeokn@gmail.com>

#     GIoU = IoU - (C - (A U B))/C
#     GIoU_Loss = 1 - GIoU
#     Args:
#     preds: [n,4] ltrb
#     targets: [n,4]
#     '''
    
#     lt_min = tf.minimum(preds[:, :,:2], targets[:, :,:2])
#     rb_min = tf.minimum(preds[:, :, 2:], targets[:, :, 2:])
#     wh_min = tf.clip_by_value(rb_min + lt_min, 0, 1e10)
#     overlap = wh_min[:, :, 0]*wh_min[:, :, 1]#[b, n]
#     area1 = (preds[:, :, 2] + preds[:, :, 0])*(preds[:, :, 3] + preds[:, :, 1])
#     area2 = (targets[:, :, 2] + targets[:, :, 0])*(targets[:, :, 3] + targets[:, :, 1])
#     union = (area1 + area2 - overlap)
#     iou = overlap/union #[b, n]  
    
#     lt_max = tf.maximum(preds[:, :, :2], targets[:, :, :2])
#     rb_max = tf.maximum(preds[:, :, 2:], targets[:, :, 2:])
#     wh_max = tf.clip_by_value((rb_max + lt_max), 0, 1e10)
#     G_area = wh_max[:, :, 0]*wh_max[:, :, 1]#[b, n]

#     giou = iou - (G_area - union)/tf.clip_by_value(G_area, 1e-10, 1e10)
#     loss = 1. - giou
#     return loss*mask

def giou_loss(preds, targets):
    '''
     # Copyright (C) 2019 * Ltd. All rights reserved.
     # author : SangHyeon Jo <josanghyeokn@gmail.com>

     GIoU = IoU - (C - (A U B))/C
     GIoU_Loss = 1 - GIoU
     Args:
     preds: [n,4] ltrb
     targets: [n,4]
    '''
    
    lt_min = tf.minimum(preds[:,:2], targets[:,:2])
    rb_min = tf.minimum(preds[:, 2:], targets[:, 2:])
#     wh_min = tf.clip_by_value(rb_min + lt_min, 0, 1e4)
    wh_min = tf.maximum(rb_min + lt_min, 0)
    overlap = wh_min[:, 0]*wh_min[:, 1]#[n]
    area1 = (preds[:, 2] + preds[:, 0])*(preds[:, 3] + preds[:, 1])
    area2 = (targets[:, 2] + targets[:, 0])*(targets[:, 3] + targets[:, 1])
    union = (area1 + area2 - overlap)
    iou = overlap/union #[n]  
    
    lt_max = tf.maximum(preds[:, :2], targets[:, :2])
    rb_max = tf.maximum(preds[:, 2:], targets[:, 2:])
#     wh_max = tf.clip_by_value((rb_max + lt_max), 0, 1e4)
    wh_max = tf.maximum(rb_max + lt_max, 0)
    G_area = wh_max[:, 0]*wh_max[:, 1]#[n]

    giou = iou - (G_area - union)/tf.maximum(G_area, 1e-10)
    loss = 1. - giou
    return loss


def compute_cls_loss_(preds, targets, class_num, mask):
    '''
    Args  
    preds: list contains five level pred [batch_size,_h,_w,class_num]
    targets: [batch_size,sum(_h*_w),class_num]
    mask: [batch_size,sum(_h*_w)]
    '''
    mask = tf.expand_dims(mask, axis = -1)
    num_pos = tf.reduce_sum(mask, axis = [1,2])#[batch_size,]
#     num_pos = tf.cast(tf.clip_by_value(num_pos, int(1), int(1e4)), tf.float32)
    num_pos = tf.cast(tf.maximum(num_pos, int(1)), tf.float32)
    b = preds[0].get_shape().as_list()[0]
    c = preds[0].get_shape().as_list()[-1]
#     print(len(preds), b, c)
    preds_ = []
    for i in range(len(preds)):
#         print(i)
        preds_.append(tf.reshape(preds[i], [b, -1, c]))
    preds = preds_
    preds = tf.concat(preds, axis = 1)
    targets = tf.one_hot(tf.cast(targets, tf.int32), class_num, axis=2)
    targets = tf.to_float(targets[:, :, 1:])
    loss = focal_loss(tf.reshape(preds, [b, -1, c]), tf.reshape(targets, [b, -1, c]), tf.squeeze(mask, axis = -1))
    return tf.reduce_sum(loss/tf.stop_gradient(tf.expand_dims(num_pos, axis = -1)), [0, 1])

def compute_cls_loss(preds, targets, class_num, mask):
    mask = tf.expand_dims(mask, axis = -1)
    num_pos = tf.reduce_sum(mask, axis = [1,2])#[batch_size,]
#     num_pos = tf.cast(tf.clip_by_value(num_pos, int(1), int(1e4)), tf.float32)
    num_pos = tf.cast(tf.maximum(num_pos, int(1)), tf.float32)
    b = preds[0].get_shape().as_list()[0]
    c = preds[0].get_shape().as_list()[-1]
    preds_ = []
    for i in range(len(preds)):
        preds_.append(tf.reshape(preds[i], [b, -1, c]))
    preds = preds_
    preds = tf.concat(preds, axis = 1)
    loss=[]
    for batch_index in range(b):
        pred_i = preds[batch_index]#[-1, 4]
        target_i = targets[batch_index]
        target_i = tf.one_hot(tf.cast(target_i, tf.int32), class_num, axis=1)
#         print(tf.shape(target_i))
        target_i = tf.squeeze(target_i[:, 1:])
        target_i = tf.to_float(target_i)
        loss.append(focal_loss_from_logits(pred_i, target_i)/tf.stop_gradient(num_pos[batch_index]))    
#     print(sess.run(loss), sess.run(num_pos))
    return loss


def compute_reg_loss(preds,targets,mask):
    '''
    Args  
    preds: list contains five level pred [batch_size,_h,_w,4]
    targets: [batch_size,sum(_h*_w),4]
    mask: [batch_size,sum(_h*_w)]
    '''

    b = preds[0].get_shape().as_list()[0]
    c = preds[0].get_shape().as_list()[-1]
    preds_reshape = []
    mask = tf.expand_dims(mask, axis = -1)#[batch_size,sum(_h*_w),1]
    # mask=targets>-1#[batch_size,sum(_h*_w),1]
    num_pos = tf.reduce_sum(mask, axis = [1,2])#[batch_size,]
#     num_pos = tf.cast(tf.clip_by_value(num_pos, int(1), int(1e4)), tf.float32)
    num_pos = tf.cast(tf.maximum(num_pos, int(1)), tf.float32)
    preds_ = []
    for i in range(len(preds)):
        preds_.append(tf.reshape(preds[i], [b, -1, c]))
    preds = preds_
    preds = tf.concat(preds, axis = 1)#[b, -1, c]
    
    loss=[]
    for batch_index in range(b):
        preds_i = preds[batch_index]#[-1, 4]
        mask_i = tf.squeeze(mask[batch_index], axis = -1)#[-1,]
        mask_i = tf.where(mask_i > 0)
        mask_i = tf.squeeze(mask_i, axis = -1)
        pred_pos_i = tf.gather(preds_i, mask_i)#[num_pos_b,]
        targets_i = targets[batch_index]
        target_pos_i=tf.gather(targets_i, mask_i)#[num_pos_b,]
#         print('xxx:', pred_pos_i, target_pos_i)
        loss_i = giou_loss(pred_pos_i, target_pos_i)
        loss_i = tf.reduce_sum(loss_i, axis = 0)/tf.stop_gradient(num_pos[batch_index])
        loss.append(loss_i)
    return loss#[batch_size,]
    


class LOSS(object):
    def __init__(self):
        self.i= 0
        
    def forward(self,inputs):
        
        '''
        inputs list
        [0]preds:  ....
        [1]targets : list contains three elements [[batch_size,sum(_h*_w),1],[batch_size,sum(_h*_w),1],[batch_size,sum(_h*_w),4]]
        '''
        preds, targets=inputs
        if cfg.cnt_branch:
            cls_logits, cnt_logits, reg_preds = preds
        else:
            cls_logits, reg_preds = preds
        class_num = cls_logits[0].get_shape().as_list()[-1]
        cls_targets, cnt_targets, reg_targets = targets
        mask_pos=tf.squeeze(tf.cast((cnt_targets>-1), tf.int32), axis = -1)# [batch_size,sum(_h*_w)]
#         print('preds, targets: ', cls_targets.get_shape().as_list())
        cls_loss=tf.reduce_mean(compute_cls_loss(cls_logits,cls_targets, class_num+1, mask_pos))#[]
        if cfg.cnt_branch:
            cnt_loss=tf.reduce_mean(compute_cnt_loss(cnt_logits,cnt_targets,mask_pos))
            reg_loss=tf.reduce_mean(compute_reg_loss(reg_preds,reg_targets,mask_pos))

#             total_loss=cls_loss+cnt_loss+reg_loss
            return cls_loss,cnt_loss,reg_loss
        else:
            reg_loss=tf.reduce_mean(compute_reg_loss(reg_preds,reg_targets,mask_pos))

#             total_loss=cls_loss+reg_loss
            return cls_loss, reg_loss
       