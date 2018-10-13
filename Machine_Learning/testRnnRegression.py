import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

batch_start=0
# 起始位置
time_steps=20
# 每个序列的输入数
batch_size=50
# 每次处理的序列树
input_size=1
# 每个输入的特征数
output_size=1
# 每个输出的结果数
cell_size=10
# 神经元个数
learning_rate=0.006
# 学习速率
def get_batch():
    '''得到输入数据'''
    global batch_start,time_steps
    # 定义全局变量
    xs=np.arange(batch_start,batch_start+time_steps*batch_size).reshape((batch_size,time_steps))/(10*np.pi)
    # 得到xs，每次取一个列表，reshape成（batch_size，time_steps）
    seq=np.sin(xs)
    # 得到输入序列sin（x）
    res=np.cos(xs)
    # 得到预测标签cos（x）
    batch_start+=time_steps
    # 每次更新下列表
    # plt.plot(xs[0, :], res[0, :], 'r', xs[0, :], seq[0, :], 'b--')
    # plt.show()
    return [seq[:,:,np.newaxis],res[:,:,np.newaxis],xs]


class LSTMRNN(object):
    '''定义LSTMRNN'''
    def __init__(self,n_steps,input_size,output_size,cell_size,batch_size):
        # 定义实例对象和属性
        self.n_steps=n_steps
        self.input_size=input_size
        self.output_size=output_size
        self.cell_size=cell_size
        self.batch_size=batch_size
        with tf.name_scope('inputs'):
            self.xs=tf.placeholder(tf.float32,[None,n_steps,input_size],name='xs')
            # 定义输入，占位符，shape为RNN的输入形状
            self.ys=tf.placeholder(tf.float32,[None,n_steps,output_size],name='ys')
        #     输出，占位符，shape为输出形状
        with tf.variable_scope('in_hidden'):
            self.add_input_layer()
        with tf.variable_scope('LSTM_cell'):
            self.add_cell()
        with tf.variable_scope('out_hidden'):
            self.add_output_layer()
        with tf.name_scope('cost'):
            self.compute_cost()
        #     定义计算损失
        with tf.name_scope('train'):
            self.train_op=tf.train.AdamOptimizer(learning_rate).minimize(self.cost)
    #         定义训练操作
    @staticmethod
    def ms_error(label,logits):
        return tf.square(tf.subtract(label,logits))
    # 静态方法 （y-y_)^2
    def _weight_variable(self,shape,name='weights'):
        '''定义weights'''
        initializer=tf.random_normal_initializer(mean=0,stddev=1)
        return tf.get_variable(shape=shape,initializer=initializer,name=name)
    def _bias_variable(self,shape,name='biases'):
        '''定义偏置'''
        initializer=tf.constant_initializer(0.1)
        return tf.get_variable(shape=shape,initializer=initializer,name=name)

    def add_input_layer(self):
        '''定义输入层'''
        l_in_x=tf.reshape(self.xs,[-1,self.input_size],name='2_2D')
        # 将输入reshape成2维数组，用于输入层的计算
        Ws_in=self._weight_variable([self.input_size,self.cell_size])
        #权重，input_size * cell_size
        bs_in=self._bias_variable([self.cell_size,])
        # 偏置 shape=[cell_size]
        with tf.name_scope('Wx_plus_b'):
            l_in_y=tf.matmul(l_in_x,Ws_in)+bs_in
        #     计算
        self.l_in_y=tf.reshape(l_in_y,[-1,self.n_steps,self.cell_size],name='2_3D')
    #     将上步的结果reshape成 cell的输入形状，3维
    def add_cell(self):
        lstm_cell=tf.contrib.rnn.BasicLSTMCell(self.cell_size,forget_bias=1.0,state_is_tuple=True)
        # 定义lstm_cell，
        with tf.name_scope('initial_state'):
            self.cell_init_state=lstm_cell.zero_state(self.batch_size,dtype=tf.float32)
        # 定义神经元初试状态
        self.cell_outputs,self.cell_final_state=tf.nn.dynamic_rnn(lstm_cell,self.l_in_y,initial_state=self.cell_init_state,time_major=False)
    #     多次处理神经元
    def add_output_layer(self):
        #shape=(batch*steps,cell_size)
        l_out_x=tf.reshape(self.cell_outputs,[-1,self.cell_size],name='2_2D')
        # 经过神经元后，连接全神经层，将上部的结果reshape成全神经元的输入形状
        Ws_out=self._weight_variable([self.cell_size,self.output_size])
        # 定义全神经层的权重
        bs_out=self._bias_variable([self.output_size])
        # 定义全神经层的偏置
        with tf.name_scope('Wx_plus_b'):
            self.pred=tf.matmul(l_out_x,Ws_out)+bs_out
    #     计算结果，最后得到的self.pred的shape是[batch*step,1]
    def compute_cost(self):
        '''计算cost，调用sequence_loss_by_example计算误差，也就是计算均方误差'''
        losses=tf.contrib.legacy_seq2seq.sequence_loss_by_example([tf.reshape(self.pred,[-1],name='reshaped_pred')],
                                                      [tf.reshape(self.ys,[-1],name='reshaped_target')],
                                                      [tf.ones([self.batch_size*self.n_steps],dtype=tf.float32)],
                                                      average_across_timesteps=True,
                                                      softmax_loss_function=self.ms_error,
                                                      name='losses')
        with tf.name_scope('average_cost'):
            self.cost=tf.div(tf.reduce_sum(losses,name='losses_sum'),self.batch_size,name='average_cost')
            tf.summary.scalar('cost',self.cost)

if __name__ == '__main__':
    model=LSTMRNN(time_steps,input_size,output_size,cell_size,batch_size)
    sess=tf.Session()
    merged=tf.summary.merge_all()
    writer=tf.summary.FileWriter('logs',sess.graph)

    init=tf.global_variables_initializer()
    sess.run(init)
    plt.ion()
    plt.show()
    for i in range(200):
        seq,res,xs=get_batch()
        if i ==0:
            feed_dict={model.xs:seq,
                       model.ys:res,
                       }
        else:
            feed_dict={
                model.xs:seq,
                model.ys:res,
                model.cell_init_state: state
            }
        _,cost,state,pred=sess.run(
            [model.train_op,model.cost,model.cell_final_state,model.pred],
            feed_dict=feed_dict
        )
        plt.plot(xs[0,:],res[0].flatten(),'r',xs[0,:],pred.flatten()[:time_steps],'b--')
        plt.ylim((-1.2,1.2))
        plt.draw()
        plt.pause(0.3)

        if i %20==0:
            print('cost:',round(cost,4))
            result=sess.run(merged,feed_dict)
            writer.add_summary(result,i)
