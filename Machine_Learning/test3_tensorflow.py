
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'#调用oracle数据库可能用到
# input1=tf.placeholder(tf.float32)
# input2=tf.placeholder(tf.float32)
# output=tf.multiply(input1,input2)
# with tf.Session() as sess:
#     print(sess.run(output,feed_dict={input1:[1,2],input2:[2,3]}))

def add_layer(inputs,in_size,out_size,n_layer,activation_function=None):
    layer_name='layer%s'%n_layer
    with tf.name_scope(layer_name):
        with tf.name_scope('W'):
            weights=tf.Variable(tf.random_normal([in_size,out_size]))
            tf.summary.histogram(layer_name+'/weights',weights)
        '''numpy random类
        numpy.random.random(size=None)　生成随机浮点数，范围0.0-1.0之间，可以通过参数size设置返回数据的size
        numpy.random.randint(low, high=None, size=None, dtype='l'）　产生随机整数，randint（3，size=2），返回的随机数都小于3
        numpy.random.normal(loc=0.0, scale=1.0, size=None) 高斯分布随机数，loc均值，scale标准差，size：抽取样本的size
        numpy.random.randn(d0, d1, …, dn)函数：从标准正态分布中返回一个(d0*d1* …* dn)维样本值
        numpy.random.rand(d0, d1, …, dn)函数，生成一个(d0*d1* …* dn)维位于[0, 1)中随机样本
        numpy.random.shuffle() 将序列的所有元素随机排序<传入参数可以是一个序列或者元组>
        numpy.random.choice(a[],size,p)可以从序列(字符串、列表、元组等)中随机选取，返回一个列表，元组或字符串的随机项
        抽取的概率p=[]其中要包含每项被抽到的概率值
        numpy.random.binomial(n,p,size=None)　二项分布采样,n个样本中，成功的概率为p，抽取size次返回抽样结果（可以用来计算概率）'''
        with tf.name_scope('b'):
            biases=tf.Variable(tf.zeros([1,out_size])+0.1)
            tf.summary.histogram(layer_name+'/biases',biases)
        #tf.zeros（）函数的参数是列表。。。numpy是元组
        with tf.name_scope('wx_plus_b'):
            wx_plus_b=tf.matmul(inputs,weights)+biases
        if activation_function is None:
            outputs=wx_plus_b
        else:
            outputs=activation_function(wx_plus_b)
        tf.summary.histogram(layer_name+'/outputs',outputs)
        return outputs


x_data=np.linspace(-1,1,300)[:,np.newaxis]
'''np.newaxis 为 numpy.ndarray（多维数组）增加一个轴,此处是把在一维数组的基础上扩展成二维数组，变成n行1列的数组
numpy 的linspace函数numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)在指定的间隔内返回均匀间隔的数字。
返回num均匀分布的样本，在[start, stop]。这个区间的端点可以任意的被排除在外。'''
noise=np.random.normal(0,0.05,x_data.shape)
y_data=np.square(x_data)-0.5+noise


with tf.name_scope('inputs'):
    xs=tf.placeholder(tf.float32,[None,1],name='x_data')
    ys=tf.placeholder(tf.float32,[None,1],name='y_data')

l1=add_layer(xs,1,10,1,activation_function=tf.nn.relu)
prediction=add_layer(l1,10,1,2,activation_function=None)

with tf.name_scope('loss'):
    loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]))
    tf.summary.scalar('loss',loss)
with tf.name_scope('training'):
    training_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)


sess=tf.Session()
merged=tf.summary.merge_all()
writer=tf.summary.FileWriter('logs/',sess.graph)

init=tf.global_variables_initializer()
sess.run(init)

# fig = plt.figure()
# ax = fig.add_subplot(1,1,1)
# ax.scatter(x_data,y_data)
# plt.ion()
# plt.show()
# for i in range(1000):
#     sess.run(training_step,feed_dict={xs:x_data,ys:y_data})
#     if i % 50 == 0:
#         try:
#             ax.lines.remove(lines[0])
#         except Exception:
#             pass
#         prediction_value = sess.run(prediction,feed_dict={xs:x_data})
#         lines = ax.plot(x_data,prediction_value,'r-',lw=2)
#         plt.pause(1)

for i in range(1000):
    sess.run(training_step,feed_dict={xs:x_data,ys:y_data})
    if i%50==0:
        result=sess.run(merged,feed_dict={xs:x_data,ys:y_data})
        # 聚合描述值
        writer.add_summary(result,i)
        #添加描述值