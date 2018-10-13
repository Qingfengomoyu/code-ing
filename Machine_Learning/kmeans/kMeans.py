from numpy import *
def loadDataSet(fileName):
    dataMat=[]
    fr=open(fileName)
    for line in fr.readlines():
        curLine=line.strip().split('\t')
        fltLine=list(map(float,curLine))
        dataMat.append(fltLine)
    return dataMat
def distEclud(vecA,vecB):
    return sqrt(sum(power(vecA-vecB,2)))
def randCent(dataSet,k):
    n=shape(dataSet)[1]

    centroids=mat(zeros((k,n)))
    for j in range(n):
        minJ=min(dataSet[:,j])

        rangeJ=float(max(dataSet[:,j])-minJ)

        centroids[:,j]=minJ+rangeJ*random.rand(k,1)

    return centroids
def kMeans(dataSet,k,distMeas=distEclud,createCent=randCent):

    m=shape(dataSet)[0]
    clusterAssment=mat(zeros((m,2)))
    centroids=createCent(dataSet,k)
    clusterChanged=True
    while clusterChanged:
        clusterChanged=False
        for i in range(m):
            minDist=inf
            minIndex=-1
            for j in range(k):
                distJI=distMeas(centroids[j,:],dataSet[i,:])
                if distJI<minDist:
                    minDist=distJI
                    minIndex=j
            if clusterAssment[i,0]!=minIndex:
                clusterChanged=True
            clusterAssment[i,:]=minIndex,minDist**2
        # print(centroids)
        for cent in range(k):
            pstInClust=dataSet[nonzero(clusterAssment[:,0]==cent)[0]]
            centroids[cent,:]=mean(pstInClust,0)
    return centroids,clusterAssment
def biKmeans(dataSet,k,distMeas=distEclud):
    # 二分K-均值聚类
    m=shape(dataSet)[0]
    clusterAssment=mat(zeros((m,2)))
    centroid0=mean(dataSet,axis=0).tolist()[0]
    centList=[centroid0]
    for j in range(m):
        clusterAssment[j,1]=distMeas(mat(centroid0),dataSet[j,:])**2
    while (len(centList))<k:
        lowestSSE=inf
        for i in range(len(centList)):
            ptsInCurrCluster=dataSet[nonzero(clusterAssment[:,0].A==i)[0],:]
            centroidMat,splitClustAss=kMeans(ptsInCurrCluster,2,distMeas)
            sseSplit=sum(splitClustAss[:,1])
            sseNoSplit=sum( clusterAssment[ nonzero(clusterAssment[:,0]!=i)[0],1])
            print('seeSplit %f,and seeNoSplit:%f'%(sseSplit,sseNoSplit))
            if (sseSplit+sseNoSplit)<lowestSSE:
                bestCentToSplit=i
                bestNewCents=centroidMat
                print(bestNewCents)
                bestClustAss=splitClustAss.copy()
                lowestSSE=sseSplit+sseNoSplit
        bestClustAss[nonzero(bestClustAss[:,0].A==1)[0],0]=len(centList)
        print(len(centList))
        bestClustAss[nonzero(bestClustAss[:,0].A==0)[0],0]=bestCentToSplit
#
        print(bestCentToSplit)
        print('the bestCentToSplit is :',bestCentToSplit)
        print('the len of bestClustAss is :',bestClustAss)
        centList[bestCentToSplit]=bestNewCents[0,:].tolist()[0]
        centList.append(bestNewCents[1,:].tolist()[0])
        clusterAssment[nonzero(clusterAssment[:,0].A==bestCentToSplit)[0],:]=bestClustAss
    return mat(centList),clusterAssment

def distSLC(vecA,vecB):
    '''根据经纬度计算地球表面的距离'''
    '''向量OA*向量OB=|OA||OB|cosθ,在直角坐标系中计算
    Ax=R*cosAw*cosAj
    Ay=R*cosAw*sinAj
    Az=R*sinAw
    
    Bx=R*cosBw*cosBj
    By=R*cosBw*sinBj
    Bz=R*sinBw
    cosθ=cosAw*cosAj*cosBw*cosBj+cosAw*sinAj*cosBw*sinBj+sinAw*sinBw
     =cosAw*cosBw(cosAj*cosBj+sinAj*sinBj)+sinAw*sinBw
     =cosAw*cosBw*cos(Aj-Bj)+sinAw*sinBw
    θ=arccos[cosAw*cosBw*cos(Aj-Bj)+sinAw*sinBw]
    弧长AB=R*arccos[cosAw*cosBw*cos(Aj-Bj)+sinAw*sinBw]
    '''
    a=sin(vecA[0,1]*pi/180)*sin(vecB[0,1]*pi/180)
    b=cos(vecA[0,1]*pi/180)*cos(vecB[0,1]*pi/180)*cos((vecB[0,0]-vecA[0,0])*pi/180)
    return 6371.0*arccos(a+b)

import matplotlib
import matplotlib.pyplot as plt




def clusterClubs(numClust=5):
    datList=[]
    fr=open('H:\IDM下载\机器学习实战pdf\MLiA_SourceCode\machinelearninginaction\Ch10/places.txt')
    for line in fr.readlines():
        lineArr=line.split('\t')
        datList.append([float(lineArr[4]),float(lineArr[3])])
    datMat=mat(datList)
    myCentroids,clustAssing=biKmeans(datMat,numClust,distMeas=distSLC)
    fig=plt.figure()
    rect=[0.1,0.1,0.8,0.8]
    scatterMarkers=['s','o','^','8','p','d','v','h','>','<']
    axprops=dict(xticks=[],yticks=[])
    ax0=fig.add_axes(rect,label='ax0',**axprops)
    imgP=plt.imread('H:\IDM下载\机器学习实战pdf\MLiA_SourceCode\machinelearninginaction\Ch10/Portland.png')
    ax0.imshow(imgP)
    ax1=fig.add_axes(rect,label='ax1',frameon=False)
    for i in range(numClust):
        pstInCurrCluster=datMat[nonzero(clustAssing[:,0].A==i)[0],:]
        markerStyle=scatterMarkers[i%len(scatterMarkers)]
        ax1.scatter(pstInCurrCluster[:,0].flatten().A[0],pstInCurrCluster[:,1].flatten().A[0],marker=markerStyle,s=90)
    ax1.scatter(myCentroids[:,0].flatten().A[0],myCentroids[:,1].flatten().A[0],marker='+',s=300)
    # plt.savefig('fig.png', bbox_inches='tight')
    plt.show()



if __name__=='__main__':
    # dataMat=loadDataSet('H:\IDM下载\机器学习实战pdf\MLiA_SourceCode\machinelearninginaction\Ch10/testSet.txt')
    #
    # dataMat=mat(dataMat)
    #
    # a=randCent(dataMat,2)
    # print(a)
    # print(distEclud(dataMat[0],dataMat[1]))
    # myCentroids,clustAssing=kMeans(dataMat,4)


    # 画聚类图，以颜色标注聚类块
    # import matplotlib.pyplot as plt
    # a=[]
    #
    # for i in range(4):
    #     listOfIndex=nonzero(clustAssing[:,0]==i)[0]
    #     a.append(listOfIndex)
    #
    # for j in range(4):
    #     if j==0:
    #        plt.plot(dataMat[a[j]][:,0],dataMat[a[j]][:,1],'or')
    #     if j==1:
    #        plt.plot(dataMat[a[j]][:,0],dataMat[a[j]][:,1],'ob')
    #     if j==2:
    #        plt.plot(dataMat[a[j]][:,0],dataMat[a[j]][:,1],'og')
    #     if j==3:
    #        plt.plot(dataMat[a[j]][:,0],dataMat[a[j]][:,1],'ok')
    # plt.show()

    # dataMat3=loadDataSet('H:\IDM下载\机器学习实战pdf\MLiA_SourceCode\machinelearninginaction\Ch10/testSet2.txt')
    # centList,myNewClustAssment=biKmeans(mat(dataMat3),3)
    # print(myNewClustAssment)
    # print(centList)
    # dataMat3=mat(dataMat3)
    # import matplotlib.pyplot as plt
    # a=[]
    #
    # for i in range(3):
    #     listOfIndex=nonzero(myNewClustAssment[:,0]==i)[0]
    #     a.append(listOfIndex)
    # print(a)
    #
    # for j in range(3):
    #     if j==0:
    #        plt.plot(dataMat3[a[j]][:,0],dataMat3[a[j]][:,1],'or')
    #     if j==1:
    #        plt.plot(dataMat3[a[j]][:,0],dataMat3[a[j]][:,1],'ob')
    #     if j==2:
    #        plt.plot(dataMat3[a[j]][:,0],dataMat3[a[j]][:,1],'og')
    #
    # plt.show()
    clusterClubs(5)