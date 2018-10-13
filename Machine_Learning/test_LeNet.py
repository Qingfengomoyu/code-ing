import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import os

inputNode=784
outputNode=10
imageSize=28
numChannels=1
numLabels=10

conv1Depth=32
conv1Size=5
conv2Depth=64
conv2Size=5
fc_size=512

def inference(input_tensor,train,regularizer):
    with tf.variable_scope('layer1_convl'):
        conv1_weights=tf.Variable(tf.truncated_normal([conv1Size,conv1Size,numChannels,conv1Depth],stddev=0.1))
        conv1_biases=tf.Variable(tf.constant(0.1,shape=[conv1Depth]))
        conv1=tf.nn.conv2d(input_tensor,conv1_weights,strides=[1,1,1,1],padding='SAME')
        relu1=tf.nn.relu(tf.nn.bias_add(conv1,conv1_biases))
    with tf.variable_scope('layer2_pool1'):
        pool1=tf.nn.max_pool(relu1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    with tf.variable_scope('layer3_conv2'):
        conv2_weights=tf.Variable(tf.truncated_normal([conv2Size,conv2Size,conv1Depth,conv2Depth],stddev=0.1))
        conv2_biases=tf.Variable(tf.constant(0.1,shape=[conv2Depth]))
        conv2=tf.nn.conv2d(pool1,conv2_weights,strides=[1,1,1,1],padding='SAME')
        relu2=tf.nn.relu(tf.nn.bias_add(conv2,conv2_biases))
    with tf.variable_scope('layer4_pool2'):
        pool2=tf.nn.max_pool(relu2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        pool2Shape=pool2.get_shape().as_list()
        nodes=pool2Shape[1]*pool2Shape[2]*pool2Shape[3]
        pool2flatten=tf.reshape(pool2,[pool2Shape[0],nodes])

    with tf.variable_scope('layer5_fc1'):
        fc1_weights=tf.Variable(tf.truncated_normal([nodes,fc_size],stddev=0.1))
        if regularizer !=None:
            tf.add_to_collection('losses',regularizer(fc1_weights))
        fc1_biases=tf.Variable(tf.constant(0.1,shape=[fc_size]))
        fc1=tf.nn.relu(tf.matmul(pool2flatten,fc1_weights)+fc1_biases)
        if train :
            fc1=tf.nn.dropout(fc1,0.5)
    with tf.variable_scope('layer6_fc2'):
        fc2_weights=tf.Variable(tf.truncated_normal([fc_size,numLabels],stddev=0.1))
        if regularizer !=None:
            tf.add_to_collection('losses',regularizer(fc2_weights))
        fc2_biases=tf.Variable(tf.constant(0.1,shape=[numLabels]))
        fc2=tf.matmul(fc1,fc2_weights)+fc2_biases
        return fc2

batchSize=100
learningRateBase=0.01
learningRateDecay=0.99
regularizationRate=0.001
trainingStep=6000
movingAverageDecay=0.99
def train(mnist):
    x=tf.placeholder(tf.float32,shape=[batchSize,imageSize,imageSize,numChannels])
    y_=tf.placeholder(tf.float32,shape=[None,outputNode])
    regularizer=tf.contrib.layers.l2_regularizer(regularizationRate)
    y=inference(x,False,regularizer)
    globalStep=tf.Variable(0,trainable=False)

    variableAverages=tf.train.ExponentialMovingAverage(movingAverageDecay,globalStep)
    variableAveragesOp=variableAverages.apply(tf.trainable_variables())
    crossEntropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
    crossEntropyMean=tf.reduce_mean(crossEntropy)
    loss=crossEntropyMean+tf.add_n(tf.get_collection('losses'))
    learningRate=tf.train.exponential_decay(learningRateBase,globalStep,mnist.train.num_examples/batchSize,learningRateDecay,staircase=True)
    trainStep=tf.train.GradientDescentOptimizer(learningRate).minimize(loss,global_step=globalStep)
    with tf.control_dependencies([trainStep,variableAveragesOp]):
        trainOp=tf.no_op(name='train')

    saver=tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(trainingStep):
            xs,ys=mnist.train.next_batch(batchSize)
            reshapedXs=np.reshape(xs,(batchSize,imageSize,imageSize,numChannels))
            _,lossValue,step=sess.run([trainOp,loss,globalStep],feed_dict={x:reshapedXs,y_:ys})
            if i%100 ==0:
                print('after %d traing steps,loss on training batch is %g'%(step,lossValue))

def main(argv=None):
    mnist=input_data.read_data_sets('MNIST_data',one_hot=True)
    train(mnist)
if __name__=='__main__':
    main()



