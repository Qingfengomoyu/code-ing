import  tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
'''dropdot的作用是再每次训练中按照一定的概率（keep_prob）选中某些神经元，是其的连接权重为0，降低训练复杂度，还可以有效减少过拟合
最后测试数据的时候应该是全连接的神经元（keep_prob=1）'''
mnist=input_data.read_data_sets('MNIST_data',one_hot=True)
#读取数据集
xs=tf.placeholder(tf.float32,[None,784])
ys=tf.placeholder(tf.float32,[None,10])
#占位符
keep_prob=tf.placeholder(tf.float32)
#定义xs，ys，以及dropout（）参数
x_image=tf.reshape(xs,[-1,28,28,1])#把数据重组成像素矩阵，[-1，28，28，1]，0位-1代表数据个数，中间两位代表，像素的长宽，
# 3位表示矩阵的通道数，此示例中只有灰色图像，只包含一个通道,彩色图有3个通道
def compute_accuracy(v_xs,v_ys):
    '''计算准确率'''
    global prediction
    y_prediction=sess.run(prediction,feed_dict={xs:v_xs,keep_prob:1})
    correct_prediction=tf.equal(tf.argmax(y_prediction,1),tf.argmax(v_ys,1))
    #tf.argmax(input,axis=None,name=None,dimension=None),argmax(y,1)代表对每行的最大数求索引，得到一个列表
    #tf.equal（x1，x2）对两个列表来说，返回一个只包含True或者False的列表，以此来代表匹配正确的个数
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    #tf.cast(x,dtype,name=None),转换类型函数
    #tf.reduce_mean(),计算这个列表中True所占的比例
    result=sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys,keep_prob:1})

    return result


def weights_variable(shape):
    '''权重定义函数'''
    initial=tf.truncated_normal(shape,stddev=0.1)
    #tf.truncated_normal(shape,mean=None,stddev=None)正态分布随机生成函数
    return tf.Variable(initial)
def biases(shape):
    '''生成偏置'''
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)
def conv2d(x,W):
    #strides设置步长，格式为四个长度的列表，且0，3位要为1
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
    #conv2d(input,filter,strides,padding,use_cudnn_on_gpu=None, name=None)
    #input：[batch,in_height,in_width,in_channels]分别为[个数，高度，宽度，通道数]
    #filter：卷积核，[filter_height,filter_width,in_channels,out_channels]分别为卷积核的[高度，宽度，输入图片的通道数，输出的featuremap通道数]
    #输出的通道数代表对单个图片来说有多少个卷积核，就会输出多少张卷积后的图像
    # strides设置步长，格式为四个长度的列表，且0，3位要为1，2，3，维代表再每一维的步长
    #padding：string类型的量，有'SAME','VALID'就是VALID只能匹配内部像素；
    # 而SAME可以在图像外部补0,从而做到只要图像中的一个像素就可以和卷积核做卷积操作,而VALID不行
   # use_cudnn_on_gpu: bool类型，是否使用cudnn加速，默认为true
    #结果返回一个Tensor，这个输出，就是我们常说的featuremap
#不管池化还是卷积，对输入为多通道的图片来说，在每个r，g，b通道上做卷积运算，然后对3个结果求和，得出最后的featuremap
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
#tf.nn.max_pool(value, ksize, strides, padding, name=None)
#value：需要池化的输入，一般池化层接在卷积层后面，所以输入通常是feature map，依然是[batch, height, width, channels]这样的shape
#ksize：池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]，因为我们不想在batch和channels上做池化，所以这两个维度设为了1
#strides：和卷积类似，窗口在每一个维度上滑动的步长，一般也是[1, stride,stride, 1]
#padding：和卷积类似，可以取'VALID' 或者'SAME'
#返回一个Tensor，类型不变，shape仍然是[batch, height, width, channels]这种形式
W_conv1=weights_variable([5,5,1,32])#第一次的卷积核：patch=5x5,insize=1,outsize=32第一次输入通道为1（灰度图像），输出32个featuremap
b_conv1=biases([32])
h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
#output=28x28x32
h_pool1=max_pool_2x2(h_conv1)
#output=14x14x32
#池化操作不改变featuremap个数，即池化操作不改变通道数
W_conv2=weights_variable([5,5,32,64])#第二次卷积核：patch=5x5,insize=32,outsize=64。输入通道变为上次卷积之后得到的32个featuremap，
# 输出变为63个featuremap
b_conv2=biases([64])
h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
#output=14x14x64
h_pool2=max_pool_2x2(h_conv2)
#output=7x7x64



W_func1=weights_variable([7*7*64,1024])
#全连接层的权重函数，输入向量的长度*输出到后一层神经元的个数
b_func1=biases([1024])
h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])
#-1和None的效果一样，既不限定输入的个数
#对全连接层fc_layer来说，输入应该是向量形式，所以要先把池化过后的图像拍平
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_func1)+b_func1)
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)

W_func2=weights_variable([1024,10])
b_func2=biases([10])
prediction=tf.nn.softmax(tf.matmul(h_fc1_drop,W_func2)+b_func2)

cross_entropy=tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))
#计算交叉熵
train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#此函数是Adam优化算法：是一个寻找全局最优点的优化算法，引入了二次方梯度校正。相比于基础SGD算法，1.不容易陷于局部优点。2.速度更快
sess=tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    batch_xs,batch_ys = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys,keep_prob:0.5})
    if i%50==0:
        print(compute_accuracy(mnist.test.images,mnist.test.labels))

