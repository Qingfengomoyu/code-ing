from numpy import *
import numpy.linalg as la
import numpy as np
np.set_printoptions(suppress=True)
# 导入外部数据
def loadExData():
    return [[0, 0, 0, 2, 2],
            [0, 0, 0, 3, 3],
            [0, 0, 0, 1, 1],
            [1, 1, 1, 0, 0],
            [2, 2, 2, 0, 0],
            [5, 5, 5, 0, 0],
            [1, 1, 1, 0, 0]]
# 导入外部数据2
def loadExData2():
    return[[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]

def ecludSim(inA,inB):
    # linglg.norm()函数用来计算向量的范数，默认为2范数，计算向量的长度
    return 1.0/(1.0+la.norm(inA-inB))
def pearsSim(inA,inB):
    if len(inA)<3:
        return 1.0
    # corrcoef()实现功能是cov（i，j）/sqrt(cov(i,i)*cov(j,j)),求ij之间的皮尔逊相关系数，返回2*2的数组，取第一行第二个
    # 这里输入是列向量
    # rowvar默认为1，行是变量，列是观察值，反之亦然
    return 0.5+0.5*corrcoef(inA,inB,rowvar=0)[0][1]

def cosSim(inA,inB):
    num=float(inA.T*inB)
    demon=la.norm(inA)*la.norm(inB)
    return 0.5+0.5*num/demon
def standEst(dataMat, user, simMeas, item):
    '''给定相似度计算方法,
    计算用户user对物品item的估计评分值'''
    n = shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    # 对每个特征
    for j in range(n):
        userRating = dataMat[user,j]
        if userRating == 0:
            continue
        # 得到对菜item和j都评过分的用户id,    dataMat[:,item]>0 表达式，得到一个列向量，数值为True或者False
        # logical_and逻辑与
        # nonzero返回非零数值的索引，组成列表，在此表征对item和j都评价过的用户id
        overLap = nonzero(logical_and(dataMat[:,item].A>0, \
                                      dataMat[:,j].A>0))[0]
        if len(overLap) == 0: similarity = 0
        else:
            # 计算物品item和j之间的相似度，选取用户对这两个物品都评分的用户分数构成物品分数向量
            similarity = simMeas(dataMat[overLap,item], dataMat[overLap,j])
        print ('the %d and %d similarity is: %f' % (item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0: return 0
    else: return ratSimTotal/simTotal

def svdEst(dataMat, user, simMeas, item):
    '''
    :param dataMat: 数据矩阵
    :param user: 用户
    :param simMeas: 相似度方法
    :param item:物品
    :return:返回用户对物品的评分
    '''

    n = shape(dataMat)[1]
    simTotal = 0.0; ratSimTotal = 0.0
    # 奇异值分解  在SVD分解之后，我们只利用包含了90%能量值的奇异值，这些奇异值会以NumPy数组的形式得以保存
    U,Sigma,VT = la.svd(dataMat)
    Sig4 = mat(eye(4)*Sigma[:4]) #arrange Sig4 into a diagonal matrix
    # xformedItems = dataMat.T * U[:,:4] * Sig4.I  #create transformed items

    xformedItems= dataMat.T * U[:, :4] * Sig4.I
    x2=mat(U[:,:4]).T*dataMat
    print(xformedItems)
    print(x2.T)
    xformedItems=x2.T
    for j in range(n):
        userRating = dataMat[user,j]
        if userRating == 0 or j==item: continue
        similarity = simMeas(xformedItems[item,:].T,\
                             xformedItems[j,:].T)
        print ('the %d and %d similarity is: %f' % (item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0: return 0
    else: return ratSimTotal/simTotal
data=loadExData()
a=mat(data)
print(a.shape)
svdEst(a,2,cosSim,0)
def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=standEst):
    '''
    :param dataMat: 数据矩阵
    :param user: 预测的用户
    :param N: 需要推荐的物品数
    :param simMeas: 相似度计算方法
    :param estMethod: 标准计算用户估分的方法
    :return: None或者在所有未评分的物品上循环，对每个未评分物品，调用estMethod方法产生预测分数，按从打到小排序，为一个list
    '''
    unratedItems = nonzero(dataMat[user,:].A==0)[1]#find unrated items
    if len(unratedItems) == 0: return 'you rated everything'
    itemScores = []
    for item in unratedItems:
        estimatedScore = estMethod(dataMat, user, simMeas, item)
        itemScores.append((item, estimatedScore))
    return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[:N]



def printMat(inMat, thresh=0.8):
    for i in range(32):
        for k in range(32):
            if float(inMat[i,k]) > thresh:
                print (1,)
            else: print( 0)
        print ('')
# 图像矩阵压缩
def imgCompress(numSV=3, thresh=0.8):
    myl = []
    for line in open(r'G:\百度云下载\01-Python课程（更新至17年4月）\Python进阶熟练班\[Python2]《机器学习实战》及源代码\machinelearninginaction\Ch14/0_5.txt').readlines():
        newRow = []
        for i in range(32):
            newRow.append(int(line[i]))
        myl.append(newRow)
    myMat = mat(myl)
    print ("****original matrix******")
    printMat(myMat, thresh)
    U,Sigma,VT = la.svd(myMat)
    SigRecon = mat(zeros((numSV, numSV)))
    for k in range(numSV):#construct diagonal matrix from vector
        SigRecon[k,k] = Sigma[k]
    reconMat = U[:,:numSV]*SigRecon*VT[:numSV,:]
    print ("****reconstructed matrix using %d singular values******" % numSV)
    printMat(reconMat, thresh)