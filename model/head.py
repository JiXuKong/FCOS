import math
import numpy as np
import config as cfg
import tensorflow as tf
from model.normalization import bn_, gn_
slim = tf.contrib.slim

'''
ClsCntRegHead类：三个预测分支，返回分支列表
'''

class ClsCntRegHead(object):
    def __init__(self, class_num, is_training, fpn_stride = cfg.strides, out_channel = 256, prior=0.01,reg_norm=False):
        self.is_training = is_training
        self.prior=prior
        self.reg_norm = reg_norm
        self.fpn_stride = fpn_stride
        self.class_num=class_num
        self.out_channel = out_channel
        
#     def normal_method(self, input_, is_training, scope):
#         return bn_(input_ , is_training, scope)
    #subnet权重共享
    def baseclassification_subnet(self, features, feature_level):
        reuse1 = tf.AUTO_REUSE
        for j in range(4):
            features = slim.conv2d(features, self.out_channel, [3,3], trainable=self.is_training,
                                   weights_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                                   biases_initializer=None,
                                   stride=1,
                                   padding="SAME",
                                   activation_fn=None,
                                   normalizer_fn= tf.identity,
                                   scope='ClassPredictionTower/conv2d_' + str(j),# + str(feature_level), 
                                  reuse=reuse1)
#             features = bn_(input_ = features, is_training = self.is_training, scope = 'ClassPredictionTower/conv2d_%d/BatchNorm/feature_%d' % (j, feature_level))
            features = tf.contrib.layers.group_norm(features, trainable=self.is_training,  scope= 'ClassPredictionTower/conv2d_%d/GroupNorm/feature_%d' % (j, feature_level))
            features = tf.nn.relu6(features)

        class_feature_output = slim.conv2d(features, (self.class_num-1), [3,3], trainable=self.is_training,
                                   weights_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                                   biases_initializer=tf.constant_initializer(-math.log((1 - self.prior)/self.prior)),
                                   stride=1,
                                   activation_fn= None,
                                   scope='ClassPredictor', 
                                   reuse=reuse1)
        
        

        return class_feature_output
    def baseregression_subnet(self, features, feature_level):    
        reuse2 = tf.AUTO_REUSE
        for j in range(4):
            features = slim.conv2d(features, self.out_channel, [3,3], trainable=self.is_training,
#                                    rate = 2 if j == 0 else 1,
                                   weights_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                                   biases_initializer=None,
                                   stride=1,
                                   padding="SAME",
                                   activation_fn=None,
                                   normalizer_fn= tf.identity,
                                   scope='BoxPredictionTower/conv2d_' + str(j),# + str(feature_level), 
                                   reuse=reuse2)
#             features = bn_(input_ = features, is_training = self.is_training, scope = 'BoxPredictionTower/conv2d_%d/BatchNorm/feature_%d' % (j, feature_level))
            features = tf.contrib.layers.group_norm(features, trainable=self.is_training, scope='BoxPredictionTower/conv2d_%d/GroupNorm/feature_%d' % (j, feature_level))
            features = tf.nn.relu6(features)
        regress_feature_output = slim.conv2d(features, 4, [3,3], trainable=self.is_training,
                                   weights_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                                   stride=1,
                                   activation_fn=None,
                                   scope='BoxPredictor', 
                                   reuse=reuse2)
        if cfg.cnt_branch:
            cnt_feature_output = slim.conv2d(features, 1, [3,3], trainable=self.is_training,
                                   weights_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                                   stride=1,
                                   activation_fn= None,
                                   scope='CntPredictor', 
                                   reuse=reuse2)
        
        if self.reg_norm:
            regress_feature_output = tf.nn.relu(regress_feature_output)
            if not self.is_training:
                regress_feature_output = regress_feature_output*self.fpn_stride[feature_level]
            if cfg.cnt_branch:
                return regress_feature_output, cnt_feature_output
            else:
                return regress_feature_output
        else:
            scale_weight = slim.model_variable(str(feature_level)+'reg_scale_',
                                           shape=[1, 1, 1, 4],
                                           initializer=tf.constant_initializer(1.0),
                                           trainable=self.is_training)
            if cfg.cnt_branch:
                return tf.exp(scale_weight*regress_feature_output), cnt_feature_output
            else:
                return tf.exp(scale_weight*regress_feature_output)
    
    def pred_subnet(self, fpn_features):
        cfeatures_ = []
        cntfeatures_ = []
        rfeatures_ = []
#         feature_shape = []
        with tf.variable_scope('WeightSharedConvolutionalBoxPredictor'):
            with slim.arg_scope([slim.conv2d], weights_regularizer=slim.l2_regularizer(1e-4)):
                for i in range(3, len(fpn_features)+3):
                    class_feature_output = self.baseclassification_subnet(fpn_features[i-3], i-3)
    #                 clas_shape = tf.shape(class_feature_outpu)
                    cfeatures_.append(class_feature_output)

    #                 cnt_shape = tf.shape(cnt_feature_output)
                    
                    if cfg.cnt_branch:
                        regress_feature_output, cnt_feature_output = self.baseregression_subnet(fpn_features[i-3], i-3)
    #                 reg_shape = tf.shape(regress_feature_output)
                        rfeatures_.append(regress_feature_output) 
                        cntfeatures_.append(cnt_feature_output)
                    else:
                        regress_feature_output = self.baseregression_subnet(fpn_features[i-3], i-3)
    #                 reg_shape = tf.shape(regress_feature_output)
                        rfeatures_.append(regress_feature_output)
                if cfg.cnt_branch:
                    return cfeatures_, cntfeatures_, rfeatures_    
                else:
                    return cfeatures_, rfeatures_