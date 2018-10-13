from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt
from os  import listdir
def createDateSet():
	group = array([[1.0,1.1],[1.1,1.0],[0,0],[0,0.1]])
	labels=['A','A','B','B']
	return group,labels
def classfy0(inX,dataSet,labels,k):
	dataSetSize=dataSet.shape[0]
	# tile(A,reps)参数都是arraylike类型，将A复制成reps[0]行，reps[1]列，
	diffMat=tile(inX,(dataSetSize,1))-dataSet
	sqDiffMat=diffMat**2
	sqDistances=sqDiffMat.sum(axis=1)
	distances=sqDistances**0.5
	# 得到距离值按从小到大排序后的索引
	sortedDistIndicies=distances.argsort()
	classCount={ }
	for i in range(k):
		# 取出索引列表中的第i项对应的label（标签值或者叫类别）
		voteIlabel=labels[sortedDistIndicies[i]]
		# 按类别的统计个数放入到字典中
		# dict.get()方法用的很好，当存在voteIlabel时，返回对应的value值，不存在时，返回0，适合用来计数
		classCount[voteIlabel]=classCount.get(voteIlabel,0)+1
	sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
	#python3下dict.items()以列表返回字典的键值对，python2下是dict.iteritems()
	return sortedClassCount[0][0]

def file2matrix(filename):
	with open(filename) as f:
		arrayOLines=f.readlines()
		numberOfLines=len(arrayOLines)
		returnMat=zeros((numberOfLines,3))
		classLabelVector=[]
		index=0
		for line in arrayOLines:
			line=line.strip()
			listFromLine=line

			listFromLine=line.split('\t')
			# split('\t')用来切割数据集中两个数字间没逗号的数据
			# listFromLine = line.split(',')
			# split(',')切割数据集中带逗号“，”的数据
			returnMat[index,:]=listFromLine[0:3]
			classLabelVector.append(int(listFromLine[-1]))
			index+=1
		return  returnMat,classLabelVector
'''Mat[:,0]就是取矩阵Mat的所有行的第0列的元素，

	Mat[:,1] 就是取所有行的第1列的元素。

	Mat[:,  m:n]即取矩阵Mat的所有行中的的第m到n-1列数据，含左不含右。

	Mat[0,:]就是取矩阵X的第0行的所有元素，

	Mat[1,:]取矩阵X的第1行的所有元素。
'''
def autoNorm(dataSet):
	minVals=dataSet.min(0)
	maxVals=dataSet.max(0)
	ranges=maxVals-minVals
	normDataSet=zeros(shape(dataSet))
	m=normDataSet.shape[0]
	normDataSet=dataSet-tile(minVals,(m,1))
	normDataSet=normDataSet/tile(ranges,(m,1))
	return normDataSet,ranges,minVals
'''
	b = a.min(para): 
	当para等于0时，b是一个1*n矩阵，是矩阵a每一列的最小值组成的矩阵；
	当para等于1时，b是一个1*m矩阵，是矩阵a每一行的最小值组成的矩阵；
	max同理！！！
	#将每列的最小值放在minVals中  
    minVals = dataSet.min(0)  
    #将每列的最大值放在maxVals中  
    maxVals = dataSet.max(0)  
    #计算可能的取值范围  
    ranges=maxVals-minVals  
    #创建新的返回矩阵  
    normDataSet = zeros(shape(dataSet))  
    #得到数据集的行数  shape方法用来得到矩阵或数组的维数  
    m = dataSet.shape[0]  
    #tile:numpy中的函数。tile将原来的一个数组minVals，扩充成了m行1列的数组  
    #矩阵中所有的值减去最小值  
    normDataSet = dataSet - tile(minVals,(m,1))  
    #矩阵中所有的值除以最大取值范围进行归一化  
    normDataSet = normDataSet/tile(ranges,(m,1))  
    #返回归一矩阵 取值范围 和最小值  
    return normDataSet,ranges,minVals  
    '''
def dataClassTest(filename,hoRatio):
	# hoRatio=0.1
	dataDataMat,datingLabels=file2matrix(filename)
	normMat,ranges,minVals=autoNorm(dataDataMat)
	m=normMat.shape[0]
	numTestVecs=int(m*hoRatio)
	errorCount=0.0
	for i in range(numTestVecs):
		classfierResult=classfy0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
		print("the classfier came back with :%d,the real answer is :%d"%(classfierResult,datingLabels[i]))
		if (classfierResult!=datingLabels[i]):
			errorCount+=1
	print("the total error rate is :%f"%(errorCount/numTestVecs))

def classfyPerson():
	resultList=['not at all','in small doses','in large doses']
	percentTats=float(input("请输入玩视频游戏所占时间百分比"))
	ffMiles=float(input("请输入每年飞行里程数："))
	iceCream=float(input("请输入每周消耗冰淇淋公升数"))
	dataDataMat,datingLabels=file2matrix("H:\IDM下载\jqxxzsfydm_pdf\MLiA_SourceCode\machinelearninginaction"
										 "\Ch02\datingTestSet2.txt")
	normMat,ranges,minVals=autoNorm(dataDataMat)
	inArr=array([ffMiles,percentTats,iceCream])
	classfierResult=classfy0((inArr-minVals)/ranges,normMat,datingLabels,3)
	print("你可能喜欢这个人的程度是：%s"%resultList[classfierResult-1])

def img2vector(filename):
	#当路径下面包含\0 \t时，会出现错误，可以用\\或者/来替代前面的\0 \t
	'''一般情况下，Python解释器会将遇到的‘\’识别为路径，会自动增加一个'\'以便和转义字符进行区分，但若遇到转义字符则不增加‘\’。
例如：上述文件名将被转换为 F:\\eclipse_workspace\\machine_learning_example\\Ch02\trainingDigits\0_38.txt。因而出错。
文件路径中若包含‘\0’、'\t' 等特殊转义字符时要特别注意。
推荐文件路径写法：
F:/eclipse_workspace/machine_learning_example/Ch02/trainningDigits/0_38.txt ，斜杠反过来了，这样就不会出现歧义了。
F:\\eclipse_workspace\\machine_learning_example\\Ch02\\trainningDigits\\0_38.txt'''
	returnVect=zeros((1,1024))
	with open(filename) as f:
		for i in range(32):
			lineStr=f.readline()
			for j in range(32):
				returnVect[0,32*i+j]=int(lineStr[j])
	return returnVect

def handwritingClassTest():
	hwlabels=[]
	trainingFileList=listdir("H:\IDM下载\jqxxzsfydm_pdf\MLiA_SourceCode\machinelearninginaction\Ch02/trainingDigits")
	m=len(trainingFileList)
	trainingMat=zeros((m,1024))
	for i in range(m):
		fileNameStr=trainingFileList[i]
		fileStr=fileNameStr.split('.')[0]
		classNumStr=int(fileStr.split('_')[0])
		hwlabels.append(classNumStr)
		trainingMat[i,:]=img2vector('H:\IDM下载\jqxxzsfydm_pdf\MLiA_SourceCode\machinelearninginaction'
							   '\Ch02/trainingDigits/%s'%fileNameStr)
	testFileList=listdir('H:\IDM下载\jqxxzsfydm_pdf\MLiA_SourceCode\machinelearninginaction\Ch02/testDigits')
	errorCount=0.0
	mTest=len(testFileList)
	for i in range(mTest):
		fileNameStr=testFileList[i]
		fileStr=fileNameStr.split('.')[0]
		classNumStr=int(fileStr.split('_')[0])
		vectorUnderTest=img2vector('H:\IDM下载\jqxxzsfydm_pdf\MLiA_SourceCode\machinelearninginaction'
								   '\Ch02/testDigits/%s'%fileNameStr)
		classfierResult=classfy0(vectorUnderTest,trainingMat,hwlabels,3)
		print('the classfier came back with:%d ,the real answer is :%d'%(classfierResult,classNumStr))
		if classfierResult!=classNumStr:
			errorCount+=1
	print("the total number of errors is : %d "%errorCount)
	print(mTest)
	print("the total error rate is %f"%(errorCount/float(mTest)))


if __name__=='__main__':
	# group,labels=createDateSet()
	# print(group)
	# print(labels)
	# print(classfy0([0,0],group,labels,3))
	dataDataMat,datingLabels=file2matrix('H:\IDM下载\jqxxzsfydm_pdf\MLiA_SourceCode\machinelearninginaction\Ch02\datingTestSet2.txt')
	print(dataDataMat)
	print(datingLabels)
	fig=plt.figure()
	ax = fig.add_subplot(111)
	# ax.scatter(dataDataMat[:,1],dataDataMat[:,2])
	ax.scatter(dataDataMat[:, 0], dataDataMat[:, 1],15.0*array(datingLabels),15.0*array(datingLabels))
	plt.show()
	normMat, ranges, minVals = autoNorm(dataDataMat)
	print(normMat)
	print(ranges)
	print(minVals)
'''figure()用来创建一个图的对象。
	
	add_subplot(x,y,z) 是这么个意思：将画布划分为x行y列，图像画在从左到右从上到下的第z块。
	
	scatter() 用来画出散点。这里它接收了4个参数：
	
	（1）横轴数据。这里是dataSet[:, 0]，也就是数据集的第1个特征（飞行里程数）
	
	（2）纵轴数据。这里是dataSet[:, 1]，也就是数据集的第2个特征（游戏百分比）
	
	（3）每个散点的标度(scalar)，从实际上看是散点的半径，或者说有多“粗”。
	这里是15 * array(datingLabels)。什么意思呢？每个散点对应一个标签datingLabel，要么1，要么2，要么3。
	把这个值乘以15，变成15,30,45，意思就是标签为1的散点的“粗度”是15，以此类推。其实就是为了在图上面好看出散点的标签。
	（4）每个散点的颜色。这里是array(datingLabels)。意思是不同的散点，按照他的标签（1或2或3）给他分配一种颜色。
	目的也是为了在图上面好看出散点的标签。
'''
