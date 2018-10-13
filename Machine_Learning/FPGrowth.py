class treeNode(object):
    def __init__(self,nameValue,numOccur,parentNode):
        self.name=nameValue
        self.count=numOccur
        # 计数值
        self.nodeLink=None
        # 节点链接，值是一个treeNode类对象，（本文中是层层嵌套的）用于链接相似的元素项
        self.parent=parentNode
        # 指向当前节点的父节点，值也是类对象（在本文中是层层嵌套的）
        self.children={ }#字典类型，里面存放当前节点的子节点（本文是可嵌套的字典）
    def inc(self,numOccur):
        self.count+=numOccur
    def display(self,ind=1):
        print("    "*ind, self.name,'',self.count)
        # 输出当前节点的名字，及对应的计数值
        for child in self.children.values():
            child.display(ind+1)
#             如果当前节点的子节点非空，继续输出

def createTree(dataSet,minSup=1):
    '''创建树函数，dataSet是字典类型的，key=交易数据集，value=出现的次数'''
    headerTable={ }
    # 字典，用于存放数据集中各个元素的出现频率
    for trans in dataSet:
        for item in trans:
            headerTable[item]=headerTable.get(item,0)+dataSet[trans]
    # dataSet是字典类型，key是trans（每条交易记录），value是交易记录的次数

    for k in list(headerTable):
        # 过滤掉小于最小支持度的元素，此处因为需要删除字典，所以要用list来取字典中的值，得到只包含频繁項元素及其出现次数的字典
        if headerTable[k] < minSup:
            del(headerTable[k])
    freqItemSet=set(headerTable.keys())
    # 将上面得到的字典转成集合类型，过滤重复元素，得到频繁项集
    if len(freqItemSet) == 0:
        return None,None
    # 如果频繁项集为空，退出，并返回None，None
    for k in headerTable:
        headerTable[k]=[headerTable[k],None]
    # 将包含频繁项的列表扩展，加入一个元素（头指针），用来连接相似元素
    retTree=treeNode('Null Set',1,None)
    # 设置初始的树
    for tranSet,count in dataSet.items():
        # 对每条交易记录及其出现的次数来进行
        localD={ }
        # 建立空字典
        for item in tranSet:
            if item in freqItemSet:
                # 如果交易记录中某个元素是频繁项
                localD[item]=headerTable[item][0]
        #         取出这个元素并把其出现的次数取出来，放入localD

        if len(localD) > 0:
            orderedItems=[v[0] for v in sorted(localD.items(),key=lambda p:p[1],reverse=True)]
            # 排序，按照localD中元素的值（在整个数据集中出现频率）高低来排，此处为降序
            # print(orderedItems)
            updateTree(orderedItems,retTree,headerTable,count)
    #         更新树
    return retTree,headerTable

def updateTree(items,inTree,headerTable,count):
    # 检查是否存在该元素（节点），如果存在则计数增加
    if items[0] in inTree.children:
        inTree.children[items[0]].inc(count)
    else:
        # 不存在则添加子节点，子节点是对象类型，其属性父节点是对象类型
        inTree.children[items[0]]=treeNode(items[0],count,inTree)
        if headerTable[items[0]][1]==None:#如果头指针为空，添加到头指针列表，
            headerTable[items[0]][1]=inTree.children[items[0]]
        else:#存在的话更新头指针列表
            updateHeader(headerTable[items[0]][1],inTree.children[items[0]])
    if len(items) >1:#如果items元素大于1个，则去掉第一个元素，将处理后的列表传给迭代调用本身函数（那么这次此函数中的items列表的第一个元素是之前的第二个元素）
        updateTree(items[1::],inTree.children[items[0]],headerTable,count)

def updateHeader(nodeToTest,targetNode):
    '''不理解'''
    #节点链接指向树中该元素项的每一个实例。
# 从头指针表的 nodeLink 开始,一直沿着nodeLink直到到达链表末尾
    while(nodeToTest.nodeLink != None):
        #如果当前指针（对象类型）的节点链接非空，（指向相似元素（也是对象））
        nodeToTest=nodeToTest.nodeLink
        #将当前指针放入头指针
    nodeToTest.nodeLink=targetNode
#     如果当前对象的节点链接为空，则指向相似元素（对象类型）


def loadSimpleDat():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat
def createInitSet(dataSet):
    retDict={ }
    for trans in dataSet:
        if frozenset(trans) in retDict:
            retDict[frozenset(trans)]+=1
        else:
            retDict[frozenset(trans)]=1
    return retDict

def ascendTree(leafNode,prefixPath):
    '''向上遍历树，参数：叶节点，前缀路径'''
    if leafNode.parent != None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent,prefixPath)
def findPrefixPath(basePat,treeNode):
    conditionPatternBase={ }
    while treeNode !=None:
        prefixPath=[]
        ascendTree( treeNode,prefixPath)
        if len(prefixPath)>1:
            conditionPatternBase[frozenset(prefixPath[1:])]=treeNode.count
        treeNode=treeNode.nodeLink
    return conditionPatternBase


#
def mineTree(inTree,headerTable,minSup,preFix,freqItemList):
    bigL=[v[0] for v in sorted(headerTable.items(),key=lambda p:str(p[1]))]

    for basePat in bigL:
        newFreqSet=preFix.copy()
        # print(newFreqSet)
        newFreqSet.add(basePat)
        # print(newFreqSet)
        freqItemList.append(newFreqSet)
        # print(freqItemList)
        condPattBases=findPrefixPath(basePat,headerTable[basePat][1])
        # print(condPattBases)
        mycondTree,myHead=createTree(condPattBases,minSup)
        # print(myHead)
        if myHead!=None:
            print('conditional tree for ',newFreqSet)
            mycondTree.display(1)
            # print('\n\n')
            mineTree(mycondTree,myHead,minSup,newFreqSet,freqItemList)

def FPGrowth(dataSet,minSup=3):
    initSet=createInitSet(dataSet)
    myFPtree,myHeaderTab=createTree(initSet,minSup)
    freqItems=[]
    mineTree(myFPtree,myHeaderTab,minSup,set([]),freqItems)
    return freqItems



if __name__=="__main__":
    # rootNode=treeNode('pyramid',9,None)
    # rootNode.children['eye']=treeNode('eye',13,None)
    # rootNode.display()
    # rootNode.children['phonenix']=treeNode('phonex',3,None)
    # rootNode.display()
    # simpleDat=loadSimpleDat()
    # initSet=createInitSet(simpleDat)
    # myFPtree,myHeaderTab=createTree(initSet,3)

    # myFPtree.display()
    # print(myHeaderTab)


    # print(findPrefixPath('x',myHeaderTab['x'][1]))
    # print(findPrefixPath('z',myHeaderTab['z'][1]))
    # print(findPrefixPath('r',myHeaderTab['r'][1]))

    # freqItems=[]
    # mineTree(myFPtree,myHeaderTab,3,set([]),freqItems)
    # print(len(freqItems))
    # freqItemList=FPGrowth(simpleDat,minSup=3)
    # print(len(freqItemList))
    fr=open('H:\IDM下载\机器学习实战pdf\MLiA_SourceCode\machinelearninginaction\Ch12\kosarak.dat')
    parseData=[line.split() for line in fr.readlines()]

    freqItemList=FPGrowth(parseData,100000)
    print(len(freqItemList))

