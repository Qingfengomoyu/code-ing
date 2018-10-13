import tensorflow as  tf
from tensorflow.examples.tutorials.mnist import input_data

#select the data
mnist=input_data.read_data_sets('MNIST_data',one_hot=True)

#hyperparamenters
learningRate=0.001
# 学习速率
trainingIters=100000
# 迭代次数
batch_size=128
# 每次处理的序列个数
n_inputs=28
# 每个序列的输入数
n_steps=28
# 每个输入的特征个数
n_hidden_units=128
# 隐藏层神经元个数
n_class=10
# 分类个数，即输出的向量列数

x=tf.placeholder(tf.float32,[None,n_steps,n_inputs])
# 输入，占位符（类型，shape）
y=tf.placeholder(tf.float32,[None,n_class])
# 结果值，占位符（类型，shape）None代表不限制输入放入个数
weights={
    'in':tf.Variable(tf.random_normal([n_inputs,n_hidden_units])),
    # 输入神经元的权重
    'out':tf.Variable(tf.random_normal([n_hidden_units,n_class]))
#     输出层的权重
}
biases={
    'in':tf.Variable(tf.constant(0.1,shape=[n_hidden_units])),
    # 输入层神经元偏置
    'out':tf.Variable(tf.constant(0.1,shape=[n_class]))
#     输出层的偏置
}
def RNN(X,weights,biases):
    #hidden_layer for input to cell
    #X:(128batch,28steps,28inputs)-->[128*28,28inputs]
    X=tf.reshape(X,[-1,n_inputs])
    # 首先转换成输入层shape，转换成None个序列中的所有值
    X_in=tf.matmul(X,weights['in'])+biases['in']
    # 计算后得到神经元的结果
    #x_in:(128batch*28steps,128hidden)-->(128batchs,28steps,128hidden)
    X_in=tf.reshape(X_in,[-1,n_steps,n_hidden_units])
    # 将结果转换shape，转成None个，多个序列的神经元的个数
    #cell
    lstm_cell=tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units,forget_bias=1.0,state_is_tuple=True)
    # LSTMCell，（神经元个数，初始化全为0，state_is_tuple=True)
    _init_state=lstm_cell.zero_state(batch_size,dtype=tf.float32)
    # 初始化隐藏的神经元，初始化为全0，（每次处理的序列数，类型）
    outputs,final_state=tf.nn.dynamic_rnn(lstm_cell,X_in,initial_state=_init_state,time_major=False)
    # 输出，新状态=多次处理RNN神经元（神经元，神经元的输入（上层的输出结果），初始化的隐藏神经元，序列是否在第一个位置）
    #hidden_layer for outputs as the final results
    results=tf.matmul(final_state[1],weights['out'])+biases['out']
    # 对于 lstm 来说, final_state可被分为(c_state, h_state).，使用h_state计算
    # outputs=tf.unstack(tf.transpose(outputs,[1,0,2]))
    # #
    # results=tf.matmul(outputs[-1],weights['out'])+biases['out']
    return results

prediction=RNN(x,weights,biases)
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
#tf.nn.softmax_cross_entropy_with_logits（）
# 第一步是先对网络最后一层的输出做一个softmax，这一步通常是求取输出属于某一类的概率，
# 第二步是softmax的输出向量[Y1，Y2,Y3...]和样本的实际标签做一个交叉熵
# 如果求loss，则要做一步tf.reduce_mean操作，对向量求均值！
train_op=tf.train.AdamOptimizer(learningRate).minimize(cost)
# 计算准确值
correct_prediction=tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    step=0
    while step*batch_size<trainingIters:
        batch_xs,batch_ys=mnist.train.next_batch(batch_size)
        # 每次处理一批数据
        batch_xs=batch_xs.reshape([batch_size,n_steps,n_inputs])
        # 把数据转换成rnn需要的shape
        sess.run(train_op,feed_dict={x:batch_xs,y:batch_ys,})
        # 运行
        if step%20==0:
            print(sess.run(accuracy,feed_dict={x:batch_xs,y:batch_ys}))
        #     每次计算准确率
        step+=1
        # 更新step，每次指向下一个batch_size