from numpy import *
# 画散点图要用array，不能是mat类型
def loadDataset(filename,delim='\t'):
    fr=open(filename)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    # map的坑，要list才能显示map后的数值，完蛋玩意
    datArr = [list(map(float,line)) for line in stringArr]
    return mat(datArr)

def pca(dataMat,topNfeet=4096):
    meanVals=mean(dataMat,axis=0)
    meanRemoved=dataMat-meanVals
    # numpy 计算协方差默认一行作为一个维度，现在设置rowvar=0，
    # 以一列作为一个维度，具体的计算不关心样本，只计算维度之间的协方差
    covMat=cov(meanRemoved,rowvar=0)
    # 取出特征值特征向量
    eigVals,eigVects=linalg.eig(mat(covMat))
    # 按特征值大小排序，返回排序后的索引值
    eigValInd=argsort(eigVals)
    # 取出n相从大到小的索引值
    eigValInd=eigValInd[:-(topNfeet+1):-1]
    # 根据索引值，取出特征值大的前N 项的特征向量，从大到小排列
    redEigVects=eigVects[:,eigValInd]
    # 矩阵乘以前最大N项特征值对应的特征向量组成的矩阵，转换到新的空间，实现压缩
    lowdDataMat=meanRemoved*redEigVects
    # 构建原矩阵，根据AP=PB,A=PB(P')P为特征值组成的矩阵，B为特征向量组成的对角矩阵
    reconMat=(lowdDataMat*redEigVects.T)+meanVals
    return lowdDataMat,reconMat
def replaceNanWithMean(filename):
    dataMat=loadDataset(filename,' ')
    numFeats=shape(dataMat)[1]

    for i in range(numFeats):
        # nonzeros(a)返回数组a中值不为零的元素的下标，
        # 它的返回值是一个长度为a.ndim(数组a的轴数)的元组,内容为不为0的数的下标
        # isnan()返回的是数组的真值，如[1,2,NaN],返回[False,False,True]
        meanVal = mean(dataMat[nonzero(~isnan(dataMat[:, i].A))[0], i])
        dataMat[nonzero(isnan(dataMat[:, i].A))[0], i] = meanVal
    return dataMat

import matplotlib.pyplot as plt
import matplotlib

dataMat = replaceNanWithMean(r'G:\百度云下载\01-Python课程（更新至17年4月）'
                      r'\Python进阶熟练班\[Python2]《机器学习实战》及源代码\machinelearninginaction\Ch13/secom.data')
meanVals = mean(dataMat, axis=0)
meanRemoved = dataMat - meanVals #remove mean
covMat = cov(meanRemoved, rowvar=0)
eigVals,eigVects = linalg.eig(mat(covMat))
print(eigVals)
eigValInd = argsort(eigVals)            #sort, sort goes smallest to largest
eigValInd = eigValInd[::-1]#reverse
sortedEigVals = eigVals[eigValInd]
total = sum(sortedEigVals)
varPercentage = sortedEigVals/total*100
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(range(1, 21), varPercentage[:20], marker='^')
plt.xlabel('Principal Component Number')
plt.ylabel('Percentage of Variance')
plt.show()