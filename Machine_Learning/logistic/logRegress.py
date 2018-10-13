from numpy  import *
def loadDataSet():
    dataMat=[]
    labelMat=[]
    fr=open('H:\IDM下载\机器学习实战pdf\MLiA_SourceCode\machinelearninginaction\Ch05/testSet.txt')
    # 路径中含有可能存在的转义字符时，前面加r'或者将/写成\(反斜杠)来避免路径出错
    for line in fr.readlines():
        lineArr=line.strip().split()
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(float(lineArr[-1]))
    return dataMat,labelMat
def sigmoid(inX):
    return 1.0/(1+ exp(-inX))
def gradAscent(dataMatIn,classLabels):
    dataMatrix=mat(dataMatIn)
    labelMat=mat(classLabels).transpose()
    m,n=shape(dataMatrix)
    alpha=0.001
    maxCycles=500
    weights=ones((n,1))
    for i in range(maxCycles):
        h=sigmoid(dataMatrix*weights)
        error=(labelMat-h)
        weights=weights+alpha*dataMatrix.transpose()*error
    return weights

def plotBestFit(weights):
    # 画图对数组有效，先把矩阵转换为数组，输入时候转换
    import matplotlib.pyplot as plt
    dataMat,labelMat=loadDataSet()
    dataArr=array(dataMat)
    n=shape(dataArr)[0]
    xcord1=[]
    ycord1=[]
    xcord2=[]
    ycord2=[]
    for i in range(n):
        if int(labelMat[i])==1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    x=arange(-3.0,3.0,0.1)
    y=(-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
def stocGradAscent0(dataMatrix,classLabels):
    m,n=shape(dataMatrix)
    alpha=0.01
    weights=ones(n)
    for i in range(m):
        h=sigmoid(sum(dataMatrix[i]*weights))
        # dataMatrix是m*n的数组，dataMatrix[i]是一个长度为n的一维数组
        error=classLabels[i]-h
        weights=weights+alpha*error*dataMatrix[i]
        # 更新weights对应的每个元素
    return weights

def stocGrandAscent1(dataMatrix,classLabels,numIter=150):
    m,n=shape(dataMatrix)
    weights=ones(n)
    for j in range(numIter):
        dataIndex=list(range(m))
        for i in range(m):
            alpha=4/(1.0+j+i)+0.01
            randomIndex=int(random.uniform(0,len(dataIndex)))
            # 从列表中任取取随机数，范围在0-列表的长度之间
            # print(randomIndex)
            h=sigmoid(sum(dataMatrix[randomIndex]*weights))
            error=classLabels[randomIndex]-h
            weights=weights+alpha*error*dataMatrix[randomIndex]
            # print(dataIndex)
            del (dataIndex[randomIndex])
    return weights

# 从疝气病预测兵马的死亡率
def classfyVector(inX,weights):
    prob=sigmoid(sum(inX*weights))
    if prob>0.5:
        return 1.0
    else:
        return 0.0

# 打开训练集和测试集，并对数据进行格式化处理
def colicTest():
    feTrain=open('H:\IDM下载\机器学习实战pdf\MLiA_SourceCode\machinelearninginaction\Ch05\horseColicTraining.txt')
    frTest=open('H:\IDM下载\机器学习实战pdf\MLiA_SourceCode\machinelearninginaction\Ch05\horseColicTest.txt')
    trainingSet=[]
    trainingLabels=[]
    for line in feTrain.readlines():
        currentLine=line.strip().split('\t')
        lineArr=[]
        for i in range(21):
            lineArr.append(float(currentLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currentLine[21]))
    trainingWeights=stocGrandAscent1(array(trainingSet),trainingLabels,500)
    errorCount=0.0
    numTestVec=0.0
    for line in frTest.readlines():
        numTestVec+=1.0
        currentLine=line.strip().split('\t')
        lineArr=[]
        for i in range(21):
            lineArr.append(float(currentLine[i]))
        if int(classfyVector(lineArr,trainingWeights))!=int(currentLine[21]):
            errorCount+=1.0
    errorRate=float(errorCount)/numTestVec
    print('the error rate of this test is :%f'%errorRate)
    return errorRate

def multiTest():
    numTest=10
    errorRate=0.0
    for i in range(numTest):
        errorRate+=colicTest()
    print('after %d iterations the average error rate is :%f'%(numTest,errorRate/float(numTest)))


