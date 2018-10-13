from numpy import *
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec=[0,1,0,1,0,1]
    return postingList,classVec

def createVocabList(dataSet):
    vocabSet=set([])
    for document in dataSet:
        vocabSet=vocabSet|set(document)
    return list(vocabSet)

# 词集模型
def setOfWord2Vec(vocabSet,inputSet):
    returnVec=[0]*len(vocabSet)
    for word in inputSet:
        if word in vocabSet:
            returnVec[vocabSet.index(word)]=1
        else:
            print("the word :%s is not in my vocabulary!"%word)
    return returnVec
#词袋模型
def bagOfWord2Vec(vocabList,inputSet):
    returnVec=[0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]+=1
    return returnVec


def trainNBO(trainMatrix,trainCategory):
    numTrainDocs=len(trainMatrix)
    numWords=len(trainMatrix[0])
    pAbusive=sum(trainCategory)/float(numTrainDocs)
    p0Num=ones(numWords)
    p1Num=ones(numWords)

    p0Denom=2.0
    p1Denom=2.0

    for i in range(numTrainDocs):
        if trainCategory[i]==1:
            p1Num+=trainMatrix[i]#向量相加，对应数字直接相加
            # p1Denom+=sum(trainMatrix[i])
        #     此处作者用的不是直接的贝叶斯公式，计算条件概率时除以总词条数？？？
            p1Denom+=1.0
        else:
            p0Num+=trainMatrix[i]
            # p0Denom+=sum(trainMatrix[i])
            p0Denom+=1.0
    p1Vect=log(p1Num/p1Denom)
    p0Vect=log(p0Num/p0Denom)
    return p0Vect,p1Vect,pAbusive




def classfyNB(vec2classify,p0Vec,p1Vec,pClass):
    p1=sum(vec2classify*p1Vec)+log(pClass)
    p0=sum(vec2classify*p0Vec)+log(1-pClass)
    if p1>p0:
        return 1
    else:
        return 0
def testingNB():
    listOfPosts,listOfClasses=loadDataSet()
    print(listOfPosts,listOfClasses)

    myVocabList=createVocabList(listOfPosts)
    print(myVocabList)

    trainMat=[]
    for postinDoc in listOfPosts:
        trainMat.append(setOfWord2Vec(myVocabList,postinDoc))
    print(array(trainMat))
    p0V,p1V,pAb=trainNBO(array(trainMat),array(listOfClasses))
    testEntry=['love','my','dalmation']
    thisDoc=array(setOfWord2Vec(myVocabList,testEntry))
    print(thisDoc)
    print("testEntry's classified as :%d"%(classfyNB(thisDoc,p0V,p1V,pAb)))
    testEntry=['stupid','garbage']
    thisDoc=array(setOfWord2Vec(myVocabList,testEntry))
    print("testEntry's classified as :%d"%(classfyNB(thisDoc,p0V,p1V,pAb)))



def textParse(bigString):
    import re
    listOfTikens=re.split(r"\W*",bigString)
    return [tok.lower() for tok in listOfTikens if len(tok)>2]
def spamTest():
    docList=[]
    classList=[]
    fullText=[]
    for i in range(1,26):
        # print(i)
        wordList=textParse(open('H:\IDM下载\机器学习实战pdf\MLiA_SourceCode\machin'
                                'elearninginaction/Ch04\email\spam/%d.txt'%i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList=textParse(open('H:\IDM下载\机器学习实战pdf\MLiA_SourceCode\machin'
                                'elearninginaction/Ch04\email\ham/%d.txt'%i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList=createVocabList(docList)
    print(vocabList)

    trainingSet=list(range(50))
    testSet=[]
    for i in range(10):
        randomIndex=int(random.uniform(0,len(trainingSet)))
        # print(randomIndex)
        testSet.append(trainingSet[randomIndex])
        del(trainingSet[randomIndex])
    trainMat=[]
    trainClass=[]
    for docIndex in trainingSet:
        trainMat.append(setOfWord2Vec(vocabList,docList[docIndex]))
        trainClass.append(classList[docIndex])
    print(array(trainMat),mat(trainMat).shape)
    print(array(trainClass))
    p0V,p1V,pSpam=trainNBO(array(trainMat),array(trainClass))
    errorCount=0
    for docIndex in testSet:
        wordVector=setOfWord2Vec(vocabList,docList[docIndex])
        if classfyNB(array(wordVector),p0V,p1V,pSpam)!= classList[docIndex]:
            errorCount+=1
    # print(errorCount)
    # print(len(testSet))
    print('the error rate is :%f'%(float(errorCount)/(len(testSet))))

spamTest()
    # RSS源分类器及高频词去除函数
def calcMostFreq(vocabList,fullText):
    import operator
    freqDict={}
    for token in vocabList:
        freqDict[token]=fullText.count(token)
    sortedFreq=sorted(freqDict.items(),key=operator.itemgetter(1),reverse=True)
    return sortedFreq[:30]
def stopWords():
    import  re
    wordList=open('H:\IDM下载\机器学习实战pdf\MLiA_SourceCode\machinelearninginaction\Ch04/stopWords.txt').read()
    listOfTokens=re.split(r"\W*",wordList)
    return [token.lower() for token in listOfTokens]

def localWords(feed1,feed0):
    import feedparser
    docList=[]
    fullText=[]
    classList=[]
    minLen=min(len(feed1['entries']),len(feed0['entries']))
    for i in range(minLen):
        wordList=textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList=textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList=createVocabList(docList)

    stopWordList=stopWords()
    for stopWord in stopWordList:
        if stopWord in vocabList:
            vocabList.remove(stopWord)
    # top30Words=calcMostFreq(vocabList,fullText)
    # for pairW in top30Words:
    #     if pairW[0] in vocabList:
    #         vocabList.remove(pairW[0])
    trainingSet=list(range(2*minLen))
    testSet=[]
    for i in range(25):
        randomIndex=int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randomIndex])
        del(trainingSet[randomIndex])

    trainMat=[]
    trainClasses=[]
    for docIndex in trainingSet:
        trainMat.append(bagOfWord2Vec(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam=trainNBO(array(trainMat),array(trainClasses))

    errorCount=0
    for docIndex in testSet:
        wordVecor=bagOfWord2Vec(vocabList,docList[docIndex])
        if classfyNB(array(wordVecor),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount+=1
    # print(errorCount)
    print('the error rate is %f'%(float(errorCount)/len(testSet)))
    return vocabList,p0V,p1V

def getTopWords(ny,sf):
    import operator
    vocabList,p0V,p1V=localWords(ny,sf)
    # print(max(p0V))
    topNY=[]
    topSF=[]

    for i in range(len(p0V)):
        if p0V[i]>-3.0:
            topSF.append((vocabList[i],p0V[i]))
        if p1V[i]>-3.0:
            topNY.append((vocabList[i],p1V[i]))
    sortedSF=sorted(topNY,key=lambda pair:pair[1],reverse=True)
    for item in sortedSF:
        print(item[0])

    sortedNY=sorted(topNY,key=lambda pair:pair[1],reverse=True)
    # key=lambda pair :pair[1] 指按照第二个元素排序，索引从0开始
    for item in sortedNY:
        print(item[0])
