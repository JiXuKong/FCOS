import tensorflow as tf
import config as cfg
from model.head import ClsCntRegHead
from model.fpn_neck import FPN
from model.backbone.resnet_ import resnet_base
from model.loss import coords_fmap2orig
from model.tool.NMS import gpu_nms

'''
FCOS类：建立FCOS前向模型
DetectHead类：预测结果后处理
    _coords2boxes：网格坐标和偏移量变换回边框
    _reshape_cat_out：返回网络坐标以及预测值
    box_nms：单个类别的非极大值抑制
    batch_nms：所有类别的非极大值抑制
    _post_process：后处理
'''


class FCOS(object):
    def __init__(self, is_training):
        self.is_training = is_training
        
    def forward(self, input_):
        end_points = resnet_base(input_, self.is_training, 'resnet_v1_50')
        fpn_features = FPN(self.is_training).forward(end_points)
        if cfg.cnt_branch:
            pred_class, pred_cnt, pred_reg = ClsCntRegHead(cfg.class_num, self.is_training).pred_subnet(fpn_features)
            return [pred_class, pred_cnt, pred_reg]
        else:
            pred_class, pred_reg = ClsCntRegHead(cfg.class_num, self.is_training).pred_subnet(fpn_features)
            return [pred_class, pred_reg]
        
    
class DetectHead(object):
    def __init__(self,score_threshold,nms_iou_threshold,max_detection_boxes_num,strides):
        self.score_threshold=score_threshold
        self.nms_iou_threshold=nms_iou_threshold
        self.max_detection_boxes_num=max_detection_boxes_num
        self.strides=strides
    def forward(self,inputs):
        '''
        inputs  list [pred_class, pred_cnt, pred_reg]  
        pred_class  list contains five [batch_size,h*w, class_num]  
        pred_cnt  list contains five [batch_size,h*w, 1]  
        pred_reg   list contains five [batch_size,h*w, 4] 
        '''
        if cfg.cnt_branch:
            cls_logits,coords = self._reshape_cat_out(inputs[0],self.strides)#[batch_size,sum(_h*_w),class_num]
            cnt_logits,_ = self._reshape_cat_out(inputs[1],self.strides)#[batch_size,sum(_h*_w),1]
            reg_preds,_ = self._reshape_cat_out(inputs[2],self.strides)#[batch_size,sum(_h*_w),4]

            cls_preds = tf.nn.sigmoid(cls_logits)
            cnt_preds = tf.nn.sigmoid(cnt_logits)
            cls_scores = tf.sqrt(cnt_preds*cls_preds)#[batch_size,sum(_h*_w)]
        else:
            cls_logits,coords = self._reshape_cat_out(inputs[0],self.strides)#[batch_size,sum(_h*_w),class_num]
            reg_preds,_ = self._reshape_cat_out(inputs[1],self.strides)#[batch_size,sum(_h*_w),4]

            cls_preds = tf.nn.sigmoid(cls_logits)
        boxes=self._coords2boxes(coords, reg_preds)#[batch_size,sum(_h*_w),4]
        
        
        _cls_scores=[]
        _cls_classes=[]
        _boxes=[]
        for batch in range(cls_logits.get_shape().as_list()[0]):
            b = tf.expand_dims(boxes[batch], axis=0)
            s = tf.expand_dims(cls_preds[batch], axis=0)
            nms_box, nms_score, nms_label = gpu_nms(b, s, 20, max_boxes=50, score_thresh=self.score_threshold, nms_thresh=self.nms_iou_threshold)
            _cls_scores.append(nms_score)
            _cls_classes.append(nms_label)
            _boxes.append(nms_box)
        _cls_scores = tf.stack(_cls_scores, axis = 0)#[batch, max_num]
        _cls_classes = tf.stack(_cls_classes, axis = 0)#[batch, max_num]
        _boxes = tf.stack(_boxes, axis = 0)#[batch, max_num, 4]
        return _cls_scores, _cls_classes, _boxes
        
    
    def _reshape_cat_out(self,inputs,strides):
        '''
        Args
        inputs: list contains five [batch_size,h, w, x]
        Returns
        out [batch_size,sum(_h*_w),c]
        coords [sum(_h*_w),2]
        '''
        batch = inputs[0].get_shape().as_list()[0]
        c = inputs[0].get_shape().as_list()[-1]
        out=[]
        coords=[]
        for pred,stride in zip(inputs,strides):
            coord = coords_fmap2orig(pred, stride)
            pred = tf.reshape(pred, [batch, -1, c])
            out.append(pred)
            coords.append(coord)
        return tf.concat(out, axis = 1), tf.concat(coords, axis =0)
            
    def _coords2boxes(self, coords, offsets): 
        '''
        coords:[h*w, 2]
        offsets:[batch, h*w, 4]ltrb
        '''
        coords_ = coords
        x1y1 = coords_[None, :, :] - offsets[:, :, :2]
        x2y2 = coords_[None, :, :] + offsets[:, :, 2:]#[batch_size,sum(_h*_w),2]
        return tf.concat([x1y1, x2y2], axis = -1)#[batch_size,sum(_h*_w),4]

    def box_nms(self, boxe,score, thr):
        boxe = tf.reshape(boxe, [-1, 4])
        score = tf.reshape(score, [-1,])
        nms_indices = tf.image.non_max_suppression(boxes=boxe,
                                                   scores=score,
                                                   max_output_size=1000,
                                                   iou_threshold=thr, name='nms_indices')
        return nms_indices
    
    def batch_nms(self, boxes, scores, idxs, iou_threshold):
        if boxes.get_shape().as_list()[0] == 0:
            return boxes
        
        # strategy: in order to perform NMS independently per class.
        # we add an offset to all the boxes. The offset is dependent
        # only on the class idx, and is large enough so that boxes
        # from different classes do not overlap
        max_coordinate = tf.reduce_max(boxes, axis=[0, 1], keepdims = False)
        offsets = tf.cast(idxs, tf.float32)*(max_coordinate + 1)
        boxes_for_nms = boxes + offsets[:, None]
        keep = self.box_nms(boxes_for_nms, scores, iou_threshold)
        return keep
    def _post_process(self, preds_topk):
        '''
        cls_scores_topk [batch_size,max_num]
        cls_classes_topk [batch_size,max_num]
        boxes_topk [batch_size,max_num,4]
        '''
        _cls_scores_post=[]
        _cls_classes_post=[]
        _boxes_post=[]
        cls_scores_topk,cls_classes_topk,boxes_topk=preds_topk
        for batch in range(cls_classes_topk.get_shape().as_list()[0]):
            mask = tf.where(cls_scores_topk[batch] >= self.score_threshold)
            _cls_scores_b = tf.gather(cls_scores_topk[batch], mask)
            _cls_classes_b = tf.gather(cls_classes_topk[batch], mask)
            _boxes_b = tf.gather(boxes_topk[batch], mask)
            nms_ind = self.batch_nms(_boxes_b, _cls_scores_b, _cls_classes_b, self.nms_iou_threshold)
            _cls_scores_post.append(tf.gather(_cls_scores_b, nms_ind))
            _cls_classes_post.append(tf.gather(_cls_classes_b, nms_ind))
            _boxes_post.append(tf.gather(_boxes_b, nms_ind))
        return tf.stack(_cls_scores_post, axis = 0), tf.stack(_cls_classes_post, axis = 0), tf.stack(_boxes_post, axis = 0)