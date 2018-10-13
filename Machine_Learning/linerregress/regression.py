from numpy import *
def loadDataSet(fileName):
    numFeat=len(open(fileName).readline().split('\t'))-1
    dataMat=[]
    labelMat=[]
    fr=open(fileName)
    for line in fr.readlines():
        lineArr=[]
        curLine=line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat
def stadardRegression(xArr,yArr):
    xMat=mat(xArr)
    yMat=mat(yArr).T
    xTx=xMat.T*xMat
    if linalg.det(xTx)==0:#linalg模块是numpy中求解和矩阵相关运算的det（）函数用来求行列式
        print('This matrix is singular,cannot do inverse')
        return
    ws=xTx.I*(xMat.T*yMat)
    #I用来表示求逆
    return ws
def lwlr(testPoint,xArr,yArr,k=1.0):
    xMat=mat(xArr)
    yMat=mat(yArr).T
    m=shape(xMat)[0]
    weights=mat(eye((m)))
    for j in range(m):
        diffMat=testPoint-xMat[j,:]
        weights[j,j]=exp(diffMat*diffMat.T/(-2.0*k**2))
    xTx=xMat.T*(weights*xMat)
    if linalg.det(xTx)==0:
        print('This matrix is singular,cannot be reverse')
        return
    ws=xTx.I*(xMat.T*(weights*yMat))
    return testPoint*ws
def lwlrTest(testArr,xArr,yArr,k=1.0):
    m=shape(testArr)[0]
    yHat=zeros(m)
    for i in range(m):
        yHat[i]=lwlr(testArr[i],xArr,yArr,k)
    return yHat
def regressionError(yArr,yHatArr):
    return ((yArr-yHatArr)**2).sum()

def ridgeRegression(xMat,yMat,lam=0.2):
    xTx=xMat.T*xMat
    denom=xTx+eye((shape(xMat)[1]))*lam
    if linalg.det(denom)==0:
        print('this matrix is singular,cannot do inverse')
        return
    ws=denom.I*(xMat.T*yMat)
    return ws

def ridgeTest(xArr,yArr):
    xMat=mat(xArr)
    yMat=mat(yArr).T
    yMean=mean(yMat,0)
    #yMean的结果是一个数，mean（）方法中有个参数，axis为0，说明对行数进行压缩，得到1*n的矩阵，axis=1，说明对列数压缩，得到m*1的数组
    yMat=yMat-yMean
    #矩阵可以直接减去行向量，矩阵的每行都减去这个行向量
    xMean=mean(xMat,0)
    xVar=var(xMat,0)#var（）求方差，有参数，axis=0，对行数压缩，即是求每列的方差，axis=1，对列数压缩，求每行的方差
    xMat=(xMat-xMean)/xVar
    #xMat矩阵的每行减去xMean这个数组
    numTestPts=30
    wMat=zeros((numTestPts,shape(xMat)[1]))
    for i in range(numTestPts):
        ws=ridgeRegression(xMat,yMat,exp(i-10))
        wMat[i,:]=ws.T
    return wMat
def regularize(xMat):
    inMat=xMat.copy()
    inMean=mean(xMat,0)
    inVar=var(xMat,0)
    inMat=(inMat-inMean)/inVar
    return inMat
def stageWise(xArr,yArr,eps=0.01,numIt=40):
    xMat=mat(xArr)
    yMat=mat(yArr).T
    yMean=mean(yArr,0)
    yMat=yMat-yMean
    xMat=regularize(xMat)
    m,n=shape(xMat)
    returnMat=zeros((numIt,n))
    ws=zeros((n,1))
    wsTest=ws.copy()
    wsMax=ws.copy()
    for i in range(numIt):
        # print('--1--------')
        print(ws.T)
        lowestError=inf
        for j in range(n):
            for sign in [-1,1]:
                wsTest=ws.copy()
                # print('=====2=====')
                # print(wsTest)
                wsTest[j]+=eps*sign
                # print('=====3=====')
                # print(wsTest[j],'=',wsTest[j],'+',eps,'*',sign)
                # print('======')
                # print(wsTest)
                yTest=xMat*wsTest
                # print('====4====')
                # print(yTest)
                rssError=regressionError(yMat.A,yTest.A)
                if rssError<lowestError:
                    lowestError=rssError
                    wsMax=wsTest
        ws=wsMax.copy()
        returnMat[i,:]=ws.T
        # print(returnMat)
    return returnMat
#以下代码无效，网址失效
from time import sleep
import json
import urllib
def searchForSet(retX,retY,setNum,yr,numPce,origprc):
    sleep(10)
    myAPIstr='get from code.google.com'
    searchURL='https://www.googleapis.com/shopping/search/v1/public/products?key=%s&country=US&q=lego+%d&alt'%(myAPIstr,setNum)

    pg=urllib.request.urlopen(searchURL)
    retDict=json.loads(pg.read())
    for i in range(len(retDict['items'])):
        try:
            currItem=retDict['items'][i]
            if currItem['product']['condition']=='new':
                newFlag=1
            else:
                newFlag=0
            listOfInv=currItem['product']['inventories']
            for item in listOfInv:
                sellingPrice=item['price']
                if sellingPrice>origprc*0.5:
                    print('%d\t%d\t%d\t%f\t%f'%(yr,numPce,newFlag,origprc,sellingPrice))
                    retX.append([yr,numPce,newFlag,origprc])
                    retY.append(sellingPrice)
        except:
            print('problem with item %d'%i)
def setDataCollect(retX,retY):
    searchForSet(retX,retY,8288,2006,800,49.99)
    searchForSet(retX, retY, 10179,2007,5195,499.99)
    searchForSet(retX, retY, 10030,2002,3096,269.99)
    searchForSet(retX, retY, 10081,2007,3428,199.99)
    searchForSet(retX, retY, 10189,2008,5922,299.99)
    searchForSet(retX, retY, 10196,2009,3263,249.99)

if __name__=='__main__':
    xArr,yArr=loadDataSet('H:\IDM下载\机器学习实战pdf\MLiA_SourceCode\machinelearninginaction\Ch08/ex0.txt')
    # print(xArr)
    # print(yArr)
    # ws=stadardRegression(xArr,yArr)
    # print(ws)
    # xMat=mat(xArr)
    # yMat=mat(yArr)
    # yHat=xMat*ws
    # print(yHat)
    # import matplotlib.pyplot as  plt
    # fig=plt.figure()
    # ax=fig.add_subplot(111)
    # ax.scatter(xMat[:,1].flatten().A[0],yMat.T[:,0].flatten().A[0])
    # xCopy=xMat.copy()
    # xCopy.sort(0)
    # yHat=xCopy*ws
    # print(yHat)
    # ax.plot(xCopy[:,1],yHat)
    # plt.show()
    # yHat=xMat*ws
    # print(corrcoef(yHat.T,yMat))#计算相关系数，将yHat转置，保证其为行向量
    #
    # print(yArr[0])
    # print(lwlr(xArr[0],xArr,yArr,1.0))


    abX,abY=loadDataSet('H:\IDM下载\机器学习实战pdf\MLiA_SourceCode\machinelearninginaction\Ch08/abalone.txt')
    #
    # yHat01=lwlrTest(abX[0:99],abX[0:99],abY[0:99],0.1)
    # yHat1=lwlrTest(abX[0:99],abX[0:99],abY[0:99],1)
    # yHat10 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 10)
    # a=regressionError(abY[0:99],yHat01.T)
    # print(a)
    # b=regressionError(abY[0:99],yHat1.T)
    # print(b)
    # c=regressionError(abY[0:99],yHat10.T)
    # print(c)
    # yHat01 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 0.1)
    # yHat1 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 1)
    # yHat10 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 10)
    # a = regressionError(abY[100:199], yHat01.T)
    # print(a)
    # b = regressionError(abY[100:199], yHat1.T)
    # print(b)
    # c = regressionError(abY[100:199], yHat10.T)
    # print(c)
    # ws=stadardRegression(abX[0:99],abY[0:99])
    # yHat=mat(abX[100:199])*ws
    # d=regressionError(abY[100:199],yHat.T.A)
    # print(d)

    # abX, abY = loadDataSet('H:\IDM下载\机器学习实战pdf\MLiA_SourceCode\machinelearninginaction\Ch08/abalone.txt')
    # ridgeWeights=ridgeTest(abX,abY)
    # import matplotlib.pyplot as  plt
    # fig=plt.figure()
    # ax=fig.add_subplot(111)
    # ax.plot(ridgeWeights)
    # plt.show()


    # a=stageWise(abX,abY,0.005,1000)
    # '''测试stageWise'''

    # xMat=mat(abX)
    # yMat=mat(abY).T
    # xMat=regularize(xMat)
    # yMeam=mean(yMat,0)
    # yMat=yMat-yMeam
    # weights=stadardRegression(xMat,yMat.T)
    # print(weights.T)

    # import matplotlib.pyplot as plt
    # fig=plt.figure()
    # ax=fig.add_subplot(111)
    # ax.plot(a)
    # plt.show()


    lgX=[]
    lgY=[]
    setDataCollect(lgX,lgY)

