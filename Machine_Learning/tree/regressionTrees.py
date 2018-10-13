from numpy import *

def loadDataSet(fileName):
    dataMat=[]
    fr=open(fileName)
    for line in fr.readlines():
        curLine=line.strip().split('\t')
        fltLine=list(map(float,curLine))
        #对python2，map（float，curLine）返回列表，python3返回迭代器，所以需要加list
        #将每行映射成浮点数
        dataMat.append(fltLine)
    return dataMat
def binSplitDataSet(dataSet,feature,value):
    mat0=dataSet[nonzero(dataSet[:,feature]>value)[0],:]
    mat1 = dataSet[nonzero(dataSet[:, feature] <= value)[0], :]
    return mat0,mat1
def regLeaf(dataSet):#returns the value used for each leaf
    '''返回叶节点的平均值'''
    return mean(dataSet[:,-1])

def regErr(dataSet):
    '''计算目标的平方误差（均方误差*总样本数）'''
    return var(dataSet[:,-1]) * shape(dataSet)[0]

def createTree(dataSet,leafType=regLeaf,errType=regErr,ops=(1,4)):
    '''数据集是numpy mat类型，可选参数，叶节点类型（树，线性），误差（树误差，线性模型误差），
    ops代表构建树需要的元组，跟误差及收敛速度有关'''
    # 切分树的特征值对应索引及切分值
    feat,val=chooseBestSplit(dataSet,leafType,errType,ops)
    if feat==None:#满足停止条件时返回叶结点值
        return val
    # 切分后赋值
    returnTree={ }
    returnTree['spInd']=feat
    returnTree['spVal']=val
    # 切分后的左右子树
    lSet,rSet=binSplitDataSet(dataSet,feat,val)
    returnTree['left']=createTree(lSet,leafType,errType,ops)
    returnTree['right']=createTree(rSet,leafType,errType,ops)
    return returnTree

def chooseBestSplit(dataSet,leafType=regLeaf,errType=regErr,ops=(1,4)):
    tolS=ops[0]#允许的误差下降值
    tolN=ops[1]#切分的最小样本数
    if len(set(dataSet[:,-1].T.tolist()[0]))==1:
        #一维矩阵经常会有tolist()[0],多维的array和matrix转换成list相同，一维的不同，
        # 一维矩阵用tolist()[0]后和一维array转化的list相同
        #1找不到好的切分特征，调用regLeaf直接生成叶结点
        return None,leafType(dataSet)
    m,n=shape(dataSet)
    S=errType(dataSet)#计算当前数据集未切分之前的误差
    bestS=inf
    bestIndex=0#初始化要切分的特征索引值为0
    bestValue=0#初始化切分的值为0
    for featIndex in range(n-1):
        for splitVal in set(dataSet[:,featIndex].T.tolist()[0]):
        # for splitVal in set(dataSet[:,featIndex]):
            mat0,mat1=binSplitDataSet(dataSet,featIndex,splitVal)
            if(shape(mat0)[0]<tolN) or (shape(mat1)[0]<tolN):
                continue
            newS=errType(mat0)+errType(mat1)
            if newS<bestS:
                bestIndex=featIndex
                bestValue=splitVal
                bestS=newS
    #2 如果切分后误差效果下降不大，则取消切分，直接创建叶结点
    if (S-bestS)<tolS:
        return None,leafType(dataSet)
    mat0,mat1=binSplitDataSet(dataSet,bestIndex,bestValue)
    if(shape(mat0)[0]<tolN) or (shape(mat1)[0]<tolN):
        # 3判断切分后子集大小，小于最小允许样本数停止切分
        return None,leafType(dataSet)
    return bestIndex,bestValue
def isTree(obj):
    return (type(obj).__name__ == "dict")
def getMean(tree):
    if isTree(tree['right']):
        tree['right']=getMean(tree['right'])
    if isTree(tree['left']):
        tree['left']=getMean(tree['left'])
    return (tree['left']+tree['right'])/2.0

def prunc(tree,testData):
    if shape(testData)[0]==0:## 确认数据集非空
        return getMean(tree)
    if ( isTree(tree['left']) or isTree(tree['right'])  ):
        # 左树或者右树是树的情况下，开始切分数据
        lSet,rSet=binSplitDataSet(testData,tree['spInd'],tree['spVal'])

    if isTree(tree['left']):
        tree['left']=prunc(tree['left'],lSet)
    if isTree(tree['right']):
        tree['right']=prunc(tree['right'],rSet)
    if (not isTree(tree['left'])) and (not isTree(tree['right'])):
        # 左树和右树都不是树的情况下，说明到达了叶节点的上一层节点，开始切分数据集
        lSet,rSet=binSplitDataSet(testData,tree['spInd'],tree['spVal'])
        # 求合并前的误差
        errorNomerge=sum(power(lSet[:,-1]-tree['left'],2))+sum(power(rSet[:,-1]-tree['right'],2))
        treeMean=(tree['left']+tree['right'])/2.0
        # 求合并后的误差
        errorMerge=sum(power(testData[:,-1]-treeMean,2))
        if errorMerge<errorNomerge:
            print('merging')
            # 返回合并后的树的均值
            return treeMean
        else:
            return tree
    else:
        return tree

def linearSolve(dataSet):
    '''线性回归算法'''
    m,n=shape(dataSet)
    X=mat(ones((m,n)))
    Y=mat(ones((m,1)))
    X[:,1:n]=dataSet[:,0:n-1]
    Y=dataSet[:,-1]
    xTx=X.T*X
    if linalg.det(xTx)==0.0:
        raise NameError('this matrix is singular,cannor be inverse \n \
        try increasing the second value of ops')
    ws=xTx.I*(X.T*Y)
    return ws,X,Y
def modelLeaf(dataSet):
    ws,X,Y=linearSolve(dataSet)
    return ws
def modelErr(dataSet):
    '''返回计算好的平方误差的和'''
    ws,X,Y=linearSolve(dataSet)
    yHat=X*ws
    return sum(power(Y-yHat,2))

def regTreeEval(model,inDat):
    return float(model)
def modelTreeEval(model,inDat):
    '''定义行向量的第一个数为1，再加上输入行的其他数据'''
    n=shape(inDat)[1]
    X=mat(ones((1,n+1)))
    X[:,1:n+1]=inDat
    return float(X*model)
def treeForeCast(tree,inData,modelEval=regTreeEval):
    if not isTree(tree):
        # 不是数结构那么tree就是之前创建树时候，不满足切分条件时返回切分值
        return modelEval(tree,inData)
    if inData[tree['spInd']]>tree['spVal']:
        # 输入行中对应的切分值的索引的那个数 大于 当前树的切分值，这时候是被分往左树的
        if isTree(tree['left']):
            # 左树是值还是树，是树的话递归调用
            return treeForeCast(tree['left'],inData,modelEval)
        else:
            # 是值直接返回
            return modelEval(tree['left'],inData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'],inData,modelEval)
        else:
            return modelEval(tree['right'],inData)

def createForeCast(tree,testData,modelEval=regTreeEval):
    m=len(testData)
    yHat=mat(zeros((m,1)))
    for i in range(m):
        yHat[i,0]=treeForeCast(tree,mat(testData[i]),modelEval)
    return yHat

if __name__=='__main__':
    # testMat=mat(eye(4))
    # print(testMat)
    # mat0,mat1=binSplitDataSet(testMat,1,0.5)
    # print(mat0)
    # print(mat1)
    # print(type(testMat))
    # print(type(testMat[:, -1].T.tolist()[0]))
    # print(testMat[:,-1].T.tolist()[0])


    # myDat=loadDataSet('H:\IDM下载\机器学习实战pdf\MLiA_SourceCode\machinelearninginaction\Ch09/ex00.txt')
    # returnTrees=createTree(mat(myDat))
    # print(returnTrees)
    # myDat1=loadDataSet('H:\IDM下载\机器学习实战pdf\MLiA_SourceCode\machinelearninginaction\Ch09/ex0.txt')
    # returnTrees1=createTree(mat(myDat1))
    # print(returnTrees1)

    # myDat=mat(myDat)
    # import matplotlib.pyplot as  plt
    # plt.plot(myDat[:,0],myDat[:,1],'ro')
    # plt.show()
    # #转换为矩阵才能画图，mmmmm
    # myDat1=mat(myDat1)
    # plt.plot(myDat1[:,1],myDat1[:,2],'ro')
    # plt.show()



    # myDat2 = loadDataSet('H:\IDM下载\机器学习实战pdf\MLiA_SourceCode\machinelearninginaction\Ch09/ex0.txt')
    # returnTrees2=createTree(mat(myDat2),ops=(0,1))
    # print(returnTrees2)
    # myDat2=loadDataSet('H:\IDM下载\机器学习实战pdf\MLiA_SourceCode\machinelearninginaction\Ch09/ex2test.txt')
    # print('============')
    # pruncedTree=prunc(returnTrees2,mat(myDat2))
    # print(pruncedTree)
    #



    # myDatExp2=loadDataSet('H:\IDM下载\机器学习实战pdf\MLiA_SourceCode\machinelearninginaction\Ch09\exp2.txt')
    # treeExp2=createTree(mat(myDatExp2),modelLeaf,modelErr,(1,10))
    # print(treeExp2)
    #
    # #画图
    # import matplotlib.pyplot as plt
    # plot输入数据是矩阵型，scatter输入数据x和y要求是一维的，不一样
    # plt.plot(mat(myDatExp2)[:,0],mat(myDatExp2)[:,1],'ro')
    # plt.show()



    trainMat=loadDataSet('H:\IDM下载\机器学习实战pdf\MLiA_SourceCode\machinelearninginaction\Ch09/bikeSpeedVsIq_train.txt')
    testMat=loadDataSet('H:\IDM下载\机器学习实战pdf\MLiA_SourceCode\machinelearninginaction\Ch09/bikeSpeedVsIq_test.txt')
    trainMat=mat(trainMat)
    testMat=mat(testMat)

    myTree=createTree(trainMat,ops=(1,20))
    yHat=createForeCast(myTree,testMat[:,0])
    a=corrcoef(yHat,testMat[:,1],rowvar=0)[0,1]
    print(a)

    myTree=createTree(trainMat,modelLeaf,modelErr,(1,20))
    yHat=createForeCast(myTree,testMat[:,0],modelTreeEval)
    b=corrcoef(yHat,testMat[:,1],rowvar=0)[0,1]
    print(b)
    ws,X,Y=linearSolve(trainMat)
    print(ws)
    yHat=zeros((shape(testMat)[0],1))
    for i in range(shape(testMat)[0]):
        yHat[i]=testMat[i,0]*ws[1,0]+ws[0,0]
    c=corrcoef(yHat,testMat[:,1],rowvar=0)[0,1]
    print(c)