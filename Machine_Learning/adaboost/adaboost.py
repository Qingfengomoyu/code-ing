from numpy import *
def loadSimData():
    dataMat=matrix([[1.0,2.1],
                    [2.0,1.1],
                    [1.3,1.0],
                    [1.0,1.0],
                    [2.0,1.0]])
    classLabels=[1.0,1.0,-1.0,-1.0,1.0]
    return dataMat,classLabels

def stumpClassfy(dataMatrix,dimen,threshVal,threshIneq):
    '''简单决策树，在此只有一个节点，此函数通过判断threshIneq值来决定数据集中某列大于某个阈值时置为-1，否则小于某阈值置-1'''
    retArray=ones((shape(dataMatrix)[0],1))

    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen]<= threshVal]=-1.0
    else:
        retArray[dataMatrix[:,dimen]> threshVal]=-1.0
    return retArray
def buildStump(dataArr,classLabels,D):
    '''找到数据集上最佳的单层决策树'''
    dataMatrix=mat(dataArr)
    labelMat=mat(classLabels).T
    m,n=shape(dataMatrix)
    numSteps=10.0#定义步数
    bestStump={}#最好的树，用字典保存
    bestClassEst=mat(zeros((m,1)))#定义预测值
    minError=inf#最小误差，初始化为正无穷大
    for i in range(n):
        rangeMin=dataMatrix[:,i].min()#第i列最小值
        rangeMax=dataMatrix[:,i].max()
        stepSize=(rangeMax-rangeMin)/numSteps#定义步长
        for j in range(-1,int(numSteps)+1):
            for inequal in ['lt','gt']:
                threshVal=(rangeMin+float(j)*stepSize)#计算后设置当前比较的阈值
                predictedVals=stumpClassfy(dataMatrix,i,threshVal,inequal)
                errArr=mat(ones((m,1)))
                errArr[predictedVals==labelMat]=0#errArr用于计算权重，预测正确则设为0
                weightedError=D.T*errArr#用来计算加权错误率，
                print("split:dim %d,thresh %.2f,thresh inequal:%s,the weighted error is :%.3f"%(i,threshVal,inequal,weightedError))
                if weightedError<minError:
                    minError=weightedError
                    bestClassEst=predictedVals.copy()
                    bestStump['dim']=i
                    bestStump['thresh']=threshVal
                    bestStump['inequal']=inequal
    return bestStump,minError,bestClassEst
def adaBoostTrainDS(dataArr,classLabels,numIt=40):
    '''基于单层决策树的adaboost训练函数'''
    weakClassArr=[]#用于存放多个单层决策树，即弱分类器
    m=shape(dataArr)[0]
    D=mat(ones((m,1))/m)
    aggClassEst=mat(zeros((m,1)))#类别估计的累计值
    for i in range(m):
        bestStump,error,classEst=buildStump(dataArr,classLabels,D)
        print('D:',D.T)
        alpha=float(0.5*log((1.0-error)/max(error,1e-16)))#α
        bestStump['alpha']=alpha
        weakClassArr.append(bestStump)
        print('classEst:',classEst.T)
        expon=multiply(-1*alpha*mat(classLabels).T,classEst)
        D=multiply(D,exp(expon))
        D=D/D.sum()
        aggClassEst+=alpha*classEst
        print('aggClassEst:',aggClassEst.T)
        aggErrors=multiply(sign(aggClassEst)!=mat(classLabels).T,ones((m,1)))#往下是计算错误率
        errorRate=aggErrors.sum()/m
        print('total error：',errorRate,'\n')
        if errorRate==0.0:
            break
    return weakClassArr,aggClassEst
def adaClassify(dataToClass,classifierArr):
    '''分类器设计函数'''
    dataMatrix=mat(dataToClass)
    m=shape(dataMatrix)[0]
    aggClassEst=mat(zeros((m,1)))#类别估计的累计值，即最终结果
    for i in range(len(classifierArr)):
        classEst=stumpClassfy(dataMatrix,classifierArr[i]['dim'],classifierArr[i]['thresh'],classifierArr[i]['inequal'])
        aggClassEst+=classifierArr[i]['alpha']*classEst
        print(aggClassEst)
    return sign(aggClassEst)
def loadDataSet(fileName):
    numFeat=len(open(fileName).readline().split('\t'))
    dataMat=[]
    labelMat=[]
    fr=open(fileName)
    for line in fr.readlines():
        lineArr=[]
        curLine=line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

def plotROC(predStrengths,classLabels):
    import matplotlib.pyplot as plt
    cursor=(1.0,1.0)#设置光标的位置
    ySum=0.0#用于计算AUG折线下面的面积
    numPostiveClassLabels=sum(array(classLabels)==1.0)#标签值为1的标签数量之和
    yStep=1/float(numPostiveClassLabels)#x轴上的步进值
    xStep=1/float(len(classLabels)-numPostiveClassLabels)
    sortedIndicies=predStrengths.argsort()
    fig=plt.figure()
    fig.clf()
    ax=plt.subplot(111)
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index]==1.0:
            delX=0
            delY=yStep
        else:
            delX=xStep
            delY=0
            ySum+=cursor[1]#此处计算面积，以为xStep都想等，所以累加y轴长度，再乘以xStep可以得到面积
        ax.plot([cursor[0],cursor[0]-delX],[cursor[1],cursor[1]-delY],c='b')
        cursor=(cursor[0]-delX,cursor[1]-delY)
    ax.plot([0,1],[0,1],'b--')#画y=x的虚线
    plt.xlabel('False Postive Rate')
    plt.ylabel('True Postive Rate')
    plt.title('ROC curve for AdaBoost Horse Colic Detection System')
    ax.axis([0,1,0,1])#圈定x轴，y轴的坐标范围
    plt.show()
    print('the Area under the curve is :',ySum*xStep)
