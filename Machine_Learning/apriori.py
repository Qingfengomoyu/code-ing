from numpy import *
def loadDataSet():
    return [ [1,3,4],[2,3,5],[1,2,3,5],[2,5]]
def createC1(dataSet):
    '''创建C1函数，C1是指把数据集中的所有元素分解开后放入一个冰冻的集合，就是说他们的类型是不可以改变的，
    这样在可以做字典的key
    构建初始候选项集的列表，即所有候选项集只包含一个元素，
    C1是大小为1的所有候选项集的集合'''
    C1=[ ]
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])

    C1.sort()
    return  list(map(frozenset,C1))

def scanD(dataSet,Ck,minSupport):
    '''第一个参数，数据集，要求是集合类型的数据集，即列表里存放每个记录的集合形式如[{1,2},{1,3,4},{2,5,6}]'''
    # 函数的功能是计算Ck中的项集在数据集合D(记录或者transactions)中的支持度,
    #     返回满足最小支持度的项集的集合，和所有项集支持度信息的字典。
    ssCnt={}
    for tid in dataSet:
        for can in Ck:
            if can.issubset(tid):
                # issubset就是子集是否包含于父集
                if  ssCnt.get(can) == None:
                    # 字典的get（）函数
                    ssCnt[can]=1
                else:
                    ssCnt[can]+=1
    numItems=float(len(dataSet))
    retList=[]
    supportData={ }
    for key in ssCnt:
        # 计算支持度
        support=ssCnt[key]/numItems
        if support>=minSupport:
            retList.insert(0,key)
            # 插入到首项
        supportData[key]=support
    return retList,supportData

def aprioriGen(Lk,k):
    '''由初始候选项集的集合Lk生成新的生成候选项集，
    k表示生成的新项集中所含有的元素个数'''
    retList=[]
    lenLk=len(Lk)
    for i in range(lenLk):
        for j in range(i+1,lenLk):
            # 此处k-2的意思是总取到这列数的倒数第二位，以保证除最后一个元素，其他元素都相等，这样算出来的集合具有唯一性
            # 否则可能出现多次相同的集合（那样之后还需要过滤）
            L1=list(Lk[i])[:k-2]
            L2=list(Lk[j])[:k-2]
            L1.sort()
            L2.sort()
            if L1==L2:
                retList.append(Lk[i]|Lk[j])
    #     添加L1和L2的并集
    return retList
def apriori(dataSet,minSupport=0.5):
    C1=createC1(dataSet)
    # 创建只含单个元素的冰冻集合的列表
    D=list(map(set,dataSet))
    # 将数据集集合化，每个交易记录（每行）都成为一个集合
    L1,supportData=scanD(D,C1,minSupport)
    # 过滤出来 满足最小支持度的单个元素的冰冻集合 的列表，以及key为各个冰冻集合和值为支持度的字典
    L=[L1]#将L1作为列表的第一个元素放入L列表中
    k=2 #下一次产生的冰冻集合 列表 是含有两个元素的
    while len(L[k-2])>0 :#当当前列表内容不为空的时候，每次k增加时候，列表取到下一个元素
        Ck=aprioriGen(L[k-2],k) #产生含k个元素的冰冻集合的列表
        Lk,supK=scanD(D,Ck,minSupport)#过滤含k个元素的满足条件的冰冻集合 ，放入列表，第二个是计算含k个元素的冰
        # 冻集合的支持度并放入字典
        supportData.update(supK)#将上步产生的字典加入到支持度字典中
        L.append(Lk)#将含k个元素的冰冻集合的列表 当作元素添加到L列表中
        k+=1
    return L,supportData

def generateRules(L,supportData,minConf=0.7):
    '''生成关联规则函数'''
    bigRuleList=[]
    for i in range(1,len(L)):
        for freqSet in L[i]:
            H1=[frozenset([item]) for item in freqSet]
            # print('===================H1================')
            # print(H1)
            if (i>1):
                rulesFromConseq(freqSet,H1,supportData,bigRuleList,minConf)
            else:
                calcConf(freqSet,H1,supportData,bigRuleList,minConf)
    return bigRuleList
def calcConf(freSet,H,supportData,brl,minconf=0.7):
    prunedH=[]
    for conseq in H:
        conf=supportData[freSet]/supportData[freSet-conseq]
        if conf >= minconf:
            print(freSet-conseq,'-->',conseq,'conf :',conf)
            brl.append((freSet-conseq,conseq,conf))
            prunedH.append(conseq)
            # print('======================测试线========================================')
            # print(prunedH)
    return prunedH
def rulesFromConseq(freqSet,H,supportData,brl,minConf=0.7):
    m=len(H[0])
    # print(m)
    if len(freqSet)> m+1 :
        Hmp1=aprioriGen(H,m+1)
        # print(Hmp1)
        Hmp1=calcConf(freqSet,Hmp1,supportData,brl,minConf)
        if len(Hmp1)>1:
            rulesFromConseq(freqSet,Hmp1,supportData,brl,minConf)



if __name__=="__main__":
    # dataSet=loadDataSet()
    # # print(dataSet)
    # # c1=createC1(dataSet)
    # # print(c1)
    # # D=list(map(set,dataSet))
    # # print(D)
    # # L1,supportData=scanD(D,c1,0.5)
    # # print(L1,supportData)
    # # l=[L1]
    # # print(l)
    # L,supportData=apriori(dataSet,minSupport=0.5)
    # # print(L)
    # # print(L[0])
    # # print(L[1])
    # # print(L[2])
    # # print(L[3])
    # # print(L)
    # # print(aprioriGen(L[0],2))
    # # L,supportData=apriori(dataSet,minSupport=0.7)
    # # print(L)
    # rules=generateRules(L,supportData,minConf=0.5)
    # print(len(rules))
    fr=open('H:\IDM下载\机器学习实战pdf\MLiA_SourceCode\machinelearninginaction\Ch11\mushroom.dat')
    mushDataSet=[line.split() for line in fr.readlines()]
    L,supportData=apriori(mushDataSet,minSupport=0.3)

    for item in L[2]:
        if item.intersection('2'):
            # 支持union(联合), intersection(交), difference(差)和sysmmetric difference(对称差集)等数学运算
            print(item)
    for item in L[1]:
        if item.intersection('2'):
            print(item)