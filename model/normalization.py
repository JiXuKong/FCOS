import tensorflow as tf
def gn_(input_, scope = 'gn', group = 8, esp=1e-3, scale = False, is_training = False, decay = 0.95):
    x = tf.transpose(input_, [0,3,1,2])
    x_shape = x.get_shape().as_list()

    x = tf.reshape(x, [-1, group, x_shape[1]//group, x_shape[2], x_shape[3]])
    mean1, var1 = tf.nn.moments(x, [2,3,4], keep_dims=True)
    ema = tf.train.ExponentialMovingAverage(decay=decay)
    #建立更新mean,var的op,并加入控制依赖
    def mean_var_with_update():
        print(1)
        ema_apply_op = ema.apply([mean1, var1])
        print(2)
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(mean1), tf.identity(var1)    
#     mean, var = tf.cond(tf.cast(False, tf.bool), mean_var_with_update,
#                         lambda: (ema.average(mean1), ema.average(var1)))
#     print(3)
    x = (x-mean1)/tf.sqrt(var1 + esp)
    if scale:
        gama = tf.get_variable(scope + 'group_gama', [x_shape[1]], initializer=tf.constant_initializer(1.0))
        beta = tf.get_variable(scope  + 'group_beta', [x_shape[1]], initializer=tf.constant_initializer(0.0))
        gama = tf.reshape(gama, [1, x_shape[1], 1, 1])
        beta = tf.reshape(beta, [1, x_shape[1], 1, 1])
        x = tf.reshape(x, [-1, x_shape[1], x_shape[2], x_shape[3]]) * gama + beta
    else:
        x = tf.reshape(x, [-1, x_shape[1], x_shape[2], x_shape[3]])
    x = tf.transpose(x, [0,2,3,1])
    return x

# def bn_()

def bn_(input_, esp=1e-3, is_training = True, decay = 0.99, scope = 'bn'):
#     x = tf.contrib.layers.batch_norm(
#         inputs = input_,
#         decay=decay,
#         epsilon=esp,
#         updates_collections=tf.GraphKeys.UPDATE_OPS,
#         is_training=is_training,
#         scope=scope)
#     return x
    x = tf.layers.batch_normalization(
            inputs=input_,
            axis=-1,
            training=is_training
        )
    return x
#     x = tf.layers.batch_normalization(
#         inputs = x,
#         momentum= decay,
#         epsilon= esp,
#         training= is_training,
#         trainable = is_training)
#     return x
        
