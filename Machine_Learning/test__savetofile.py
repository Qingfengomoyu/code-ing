import tensorflow as tf
import numpy as np

# W=tf.Variable([[1,2,3],[3,4,5]],dtype=tf.float32,name='weights')
# b=tf.Variable([[4,6,8]],dtype=tf.float32,name='biases')
# init=tf.global_variables_initializer()
# saver=tf.train.Saver()
#
# with tf.Session() as sess:
#     sess.run(init)
#     saver_path=saver.save(sess,'my_net/save_net.ckpt')
#     print('save to path:',saver_path)

#restore variables
# 目前tensorflow只能保存变量，还不能保存网络结构，再保存变量的时候需要先定义变量的shape和类型
#恢复变量时首先要定义成相同的shape和类型，然后再执行恢复操作restore
#重新定义相同的神经网络框架和数据shape及类型
W=tf.Variable(np.arange(6).reshape((2,3)),dtype=tf.float32,name='weights')
b=tf.Variable(np.arange(3).reshape((1,3)),dtype=tf.float32,name='biases')
#not need init step
saver =tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess,'my_net/save_net.ckpt')
    print('weights:',sess.run(W))
    print('biases:',sess.run(b))