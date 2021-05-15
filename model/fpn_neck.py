#fpn
#上采样采用最近邻插值
#fpn通道数=256，先使用1x1卷积变换得到相同的通道数
#fpn后再使用3x3卷积平滑一下
#最后生成p6,p7，只有p6到p7之间用relu
import tensorflow as tf
from model.normalization import bn_, gn_
slim = tf.contrib.slim

'''
FPN类：
    #fpn
    #上采样采用最近邻插值
    #fpn通道数=256，先使用1x1卷积变换得到相同的通道数
    #fpn后再使用3x3卷积平滑一下
    #最后生成p6,p7，只有p6到p7之间用relu
'''

class FPN(object):
    def __init__(self, is_training, features=256, initlizer=tf.truncated_normal_initializer(mean=0.0, stddev=0.03)):
        self.features = features
        self.initlizer = initlizer
        self.is_training = is_training
        
#     def normal_method(self, input_, is_training, scope):
#         return bn_(input_ , is_training, scope)
    
    def upsample_layer(self, inputs, out_shape, scope):
        with tf.name_scope(scope):
            new_height, new_width = out_shape[0], out_shape[1]
#             print(inputs.get_shape().as_list())
            channels = tf.shape(inputs)[3]
            batch_size = tf.shape(inputs)[0]
            rate = 2
            x = tf.reshape(inputs, [batch_size, new_height//rate, 1, new_width//rate, 1, channels])
            x = tf.tile(x, [1, 1, rate, 1, rate, 1])
            x = tf.reshape(x, [batch_size, new_height, new_width, channels])
            return x
        
    def forward(self, end_points):
        with tf.variable_scope('FeatureExtractor/resnet_v1_50/fpn'):
            with slim.arg_scope([slim.conv2d], weights_regularizer=slim.l2_regularizer(1e-4)):
                for level in range(5,2,-1):#5,4,3
                    print(end_points['p'+str(level)].get_shape().as_list())
                    net = slim.conv2d(end_points['p'+str(level)], self.features, [1, 1], trainable=self.is_training,
                                        weights_initializer=self.initlizer,
                                        activation_fn=None,
                                        scope='projection_'+str(level-2))
#                     end_points['p'+str(level)] = bn_(input_ = net, is_training = self.is_training, scope = 'projection_'+str(level-2)+'/BatchNorm')
#                     end_points['p'+str(level)] = net
                    end_points['p'+str(level)] = tf.contrib.layers.group_norm(net, trainable=self.is_training, scope='projection_'+str(level-2)+'/GroupNorm')
                #p5, p4的上采样相加
                for level in range(5,3,-1):
                    plevel_up = self.upsample_layer(end_points['p'+str(level)], [tf.shape(end_points['p'+str(level-1)])[1],
                                                                 tf.shape(end_points['p'+str(level-1)])[2]],
                                                                 'p'+str(level)+'_upsample')
                    end_points['p'+str(level-1)] = tf.add(end_points['p'+str(level-1)], plevel_up, 'fuse_p'+str(level-1))


                for level in range(5,2,-1):
                    net = slim.conv2d(end_points['p'+str(level)], self.features, [3, 3], trainable=self.is_training,
                                    weights_initializer=self.initlizer,
                                    biases_initializer=None,
                                    activation_fn=None,
                                    padding="SAME",
                                    stride=1,
                                    scope='smoothing_'+str(level-2))
#                     end_points['p'+str(level)] = bn_(input_ = net, is_training = self.is_training, scope = 'smoothing_'+str(level-2)+'/BatchNorm')
                    end_points['p'+str(level)] = tf.contrib.layers.group_norm(net, trainable=self.is_training, scope='smoothing_'+str(level-2)+'/GroupNorm')
#                     end_points['p'+str(level)] = net

                p6 = slim.conv2d(end_points['p'+str(5)], 256, [3, 3], trainable=self.is_training,
                                    weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.03),
                                    biases_initializer=None,
                                    activation_fn=None,
                                    padding="SAME",
                                    stride=2,
                                    scope='bottom_up_block5')
#                 p6 = bn_(input_ = p6, is_training = self.is_training, scope = 'bottom_up_block5/BatchNorm')
                p6 = tf.contrib.layers.group_norm(p6, trainable=self.is_training, scope='bottom_up_block5/GroupNorm')
                net = tf.nn.relu6(p6)
                p7 = slim.conv2d(net, 256, [3, 3], trainable=self.is_training,
                                weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.03),
                                biases_initializer=None,
                                padding="SAME",
                                stride=2,
                                activation_fn=None,
                                scope='bottom_up_block6')
#                 p7 = bn_(input_ = p7, is_training = self.is_training, scope = 'bottom_up_block6/BatchNorm')
                p7 = tf.contrib.layers.group_norm(p7, trainable=self.is_training, scope='bottom_up_block6/GroupNorm')
                p3 = end_points['p3']
                p4 = end_points['p4']
                p5 = end_points['p5']

                return [p3, p4, p5, p6, p7]