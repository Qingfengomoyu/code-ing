import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
import matplotlib
np.random.seed(0)
X,y=sklearn.datasets.make_moons(200,noise=0.2)
plt.scatter(X[:,0],X[:,1],s=40,c=y,cmap=plt.cm.Spectral)#c是一个列表，实际上是plt.cm.Spectral()的一个参数，每个值代表一种颜色
#plt.cm.Spectral(np.arange(5))将生成5中不同的颜色
clf=sklearn.linear_model.LogisticRegressionCV()
clf.fit(X,y)

#绘制决策边界函数
def plot_decision_boundary( predict_func):
    x_min,x_max=X[:,0].min()-.5,X[:,0].max()+.5
    y_min,y_max=X[:,1].min()-.5,X[:,1].max()+.5
    h=0.01
    xx,yy=np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))
    # 生成矩阵是size（a）*size（b），x以行扩展size（b）得到xx，y转置后乘以size（a）行得到yy
    Z=predict_func(np.c_[xx.ravel(),yy.ravel()])#np.c_按行连接,np.r_以列来连接
    Z=Z.reshape(xx.shape)
    print(Z)
    plt.contourf(xx,yy,Z,cmap=plt.cm.Spectral)
    plt.scatter(X[:,0],X[:,1],c=y,cmap=plt.cm.Spectral)

plot_decision_boundary(lambda x:clf.predict(x))
plt.title('Logistic Regression')

num_examples=len(X)
# 训练集规模
nn_input_dim=2
# 输入层维度
nn_output_dim=2
# 输出层维度
epsilon=0.01
# 梯度下降的学习速率
reg_lambda=0.01#规范化强度


def calculate_loss(model):
    W1,b1,W2,b2=model['W1'],model['b1'],model['W2'],model['b2']
    # 用于计算预测值的前向传播
    z1=X.dot(W1)+b1
    a1=np.tanh(z1)
    z2=a1.dot(W2)+b2
    exp_scores=np.exp(z2)
    probs=exp_scores/np.sum(exp_scores,axis=1,keepdims=True)
    #计算损失
    corect_logprobs=-np.log(probs[range(num_examples),y])
    data_loss=np.sum(corect_logprobs)
    data_loss+=reg_lambda/2*(np.sum(np.square(W1))+np.sum(np.square(W2)))
    return 1./num_examples*data_loss

def predict(model,x):
    '''一个用于计算输出的辅助函数。它会通过定义好的前向传播方法来返回拥有最大概率的类别。'''
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    z1=x.dot( W1)+b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs,axis=1)

def build_model(nn_hdim,num_passes=20000,print_loss=False):
    '''训练神经网络的函数。它会使用我们之前找到的后向传播导数来进行批量梯度下降运算。'''
    #nn_hdim：隐藏层中的节点数
    #num_passes：
    #print_loss:如果返回True，每一千次迭代打印一次损失
    #用随机数初始化参数，我们需要学习这些参数
    np.random.seed(0)
    W1=np.random.randn(nn_input_dim,nn_hdim)/np.sqrt(nn_input_dim)
    b1=np.zeros((1,nn_hdim))
    W2=np.random.randn(nn_hdim,nn_output_dim)/np.sqrt(nn_hdim)
    b2=np.zeros((1,nn_output_dim))
    #这是用于最终反馈结果的变量
    model = { }
    #批量梯度下降
    for i in range(0, num_passes):
        #前向传播
        # Forward propagation
        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # Backpropagation后向传播
        delta3 = probs
        delta3[range(num_examples), y] -= 1
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)
        #梯度下降参数更新
        # Add regularization terms (b1 and b2 don't have regularization terms)
        dW2 += reg_lambda * W2
        dW1 += reg_lambda * W1

        # Gradient descent parameter update梯度下降参数更新
        W1 += -epsilon * dW1
        b1 += -epsilon * db1
        W2 += -epsilon * dW2
        b2 += -epsilon * db2

        # Assign new parameters to the model将新的参数赋到model
        model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

        # Optionally print the loss.选择性打印损失
        # This is expensive because it uses the whole dataset, so we don't want to do it too often.
        if print_loss and i % 1000 == 0:
            print("Loss after iteration %i: %f" % (i, calculate_loss(model)))

    return model

#创建一个隐藏层有三个维度的模型
model = build_model(3, print_loss=True)

# Plot the decision boundary绘制决策边界
plot_decision_boundary(lambda x: predict(model, x))
plt.title("Decision Boundary for hidden layer size 3")

# %% 14
plt.figure(figsize=(16, 32))
hidden_layer_dimensions = [1, 2, 3, 4, 5, 20, 50]
for i, nn_hdim in enumerate(hidden_layer_dimensions):
    plt.subplot(5, 2, i + 1)
    plt.title('Hidden Layer size %d' % nn_hdim)
    model = build_model(nn_hdim)
    plot_decision_boundary(lambda x: predict(model, x))
plt.show()




