import matplotlib.pyplot as plt
from pylab import *
import os
mpl.rcParams['font.sans-serif'] = ['SimHei']

decisionNode=dict(boxstyle="sawtooth",fc="0.8")
leafNode=dict(boxstyle="round4",fc="0.8")
arrow_args=dict(arrowstyle="<-")

def plotNode(nodeTxt,centerPt,parentPt,nodeType):
	createPlot.ax1.annotate(nodeTxt,xy=parentPt,xycoords='axes fraction',xytext=centerPt,
							textcoords='axes fraction',va="center",ha="center",
							bbox=nodeType,arrowprops=arrow_args)
# 	centerPt是终点坐标，parentPt是起点坐标，
# 添加注释plt.annotate(label,xy=(x,y),xytext=(5,2),textcoords='offset points',ha='right',va='bottom')
#
# 此处 ha='right'点在注释右边（right,center,left），va='bottom'点在注释底部('top', 'bottom', 'center', 'baseline')
#
# ha有三个选择：right,center,left
#
# va有四个选择：'top', 'bottom', 'center', 'baseline'
# annotate括号中，nodeTxt是文本内容，xy是起点，xytext是终点，bbox是文本框类型，arrowprops是箭头类型，调用完后相当于完成了对树根的绘制
def createPlot():
	fig=plt.figure(1,facecolor='white')
	fig.clf()
	createPlot.ax1=plt.subplot(111,frameon=False)
	plotNode("决策节点",(0.5,0.1),(0.1,0.5),decisionNode)
	plotNode("叶节点",(0.8,0.1),(0.3,0.8),leafNode)
	plt.show()


def getNumLeafs(myTree):
	numLeafs=0
	firstStr=list(myTree.keys())[0]
	secondDict=myTree[firstStr]
	for key in secondDict.keys():
		if type(secondDict[key]).__name__=='dict':
			numLeafs+=getNumLeafs(secondDict[key])
		else:
			numLeafs+=1
	return numLeafs
# 计算树的叶子数目，递归函数，判断字典的value值是否是字典，如果是，则递归调用下一层，如果不是，则叶子数加1
#把树转换成关键字列表，此时列表中只有一个关键字，因为是第一个分支点
def getTreeDepth(myTree):
	maxDepth=0
	firstStr=list(myTree.keys())[0]
	secondDict=myTree[firstStr]
	for key in secondDict.keys():
		if type(secondDict[key]).__name__=='dict':
			thisDepth=1+getTreeDepth(secondDict[key])
		else:
			thisDepth=1
		if thisDepth>maxDepth:
			maxDepth=thisDepth
	return maxDepth
# 计算树的高度，即层数，递归函数，判断字典的value值是否为字典，如果是，当前树的高度加1再加递归调用下层数返回的结果，若不是，高度就为1
# 判断完当前节点下所有数之后，找出最大值，再去判断同层决策节点下树的高度（如果有），最后返回最大值
def retrieveTree(i):
	listOfTree=[{'no surfacing':{0:'no',1:{'flippers':{0:'no',1:'yes'}}}}]
	return listOfTree

def plotMidText(cntrPt,parentPt,txtString):
	xMid=(parentPt[0]-cntrPt[0])/2+cntrPt[0]
	yMid=(parentPt[1]-cntrPt[1])/2+cntrPt[1]
	createPlot.ax1.text(xMid,yMid,txtString)
# 在连线上标出分类的 特征值
def plotTree(myTree,parentPt,nodeTxt):
	numLeafs=getNumLeafs(myTree)
	depth=getTreeDepth(myTree)
	firstStr=list(myTree.keys())[0]
	cntrPt = (plotTree.xOff +  0.5/plotTree.totalW+float(numLeafs) / 2 / plotTree.totalW, plotTree.yOff)
#cntrPt应该是centerPt的缩写，指树根节点，决策节点的x坐标，第一次调用时，计算树根x坐标：xoff是-0.5*半叶距，先加回来，
# 再加上所有叶距之和的一半，计算决策节点时，xoff是上一次绘制叶节点的x坐标（上一次绘制的叶子不在当前树上
	# ），加上当前树的叶数占总叶数的比例的一半，再加上半个叶距
	# x轴的坐标由前一个位置确定当前位置，第一次是由初始位置确定
	plotMidText(cntrPt,parentPt,nodeTxt)

	plotNode(firstStr,cntrPt,parentPt,decisionNode)
	secondDict=myTree[firstStr]
	plotTree.yOff=plotTree.yOff-1/plotTree.totalD
	# 树的深度往下走一级，树的深度不计算树根，y轴被分为plotTree.totalD，每层高度1.0 / plotTree.totalD
	for key in secondDict.keys():
		if type(secondDict[key]).__name__=='dict':
			plotTree(secondDict[key],cntrPt,str(key))
		else:
			plotTree.xOff=plotTree.xOff+1.0/plotTree.totalW
			# #如果不是字典，那肯定是一个节点，这个节点的x坐标位置距离上一个节点1.0 / plotTree.totalW
			plotNode(secondDict[key],(plotTree.xOff,plotTree.yOff),cntrPt,leafNode)
			plotMidText((plotTree.xOff,plotTree.yOff),cntrPt,str(key))
	plotTree.yOff=plotTree.yOff+1/plotTree.totalD

def createPlot(inTree):
	fig=plt.figure(1,facecolor='white')
	fig.clf()
	axprops=dict(xticks=[],yticks=[])

	# 去除坐标轴显示,也可以选择显示哪些点，如plt.xticks([5,6]),或者ax1.set_xticks([5,6])
	createPlot.ax1=plt.subplot(111,frameon='False',**axprops)
	# createPlot是定义的函数名，是一个对象，只要是对象就可以定义公共属性，createPlot.ax1中的.ax1就是一个公共属性
	plotTree.totalW=getNumLeafs(inTree)
	# #plotTree是个函数，函数是对象可随时添加公共属性，totalW就是加入的一个属性，
	# plotTree.totalW为叶节点的总数
	plotTree.totalD=getTreeDepth(inTree)
	# #plotTree.totalD树的深度，这两个是不变的全局变量，就表示整个树的深度和宽度
	plotTree.xOff=-0.5/plotTree.totalW
	plotTree.yOff=1.0
	plotTree(inTree,(0.5,1.0),'')
	plt.show()
# 如果要显示没有坐标没有边框的图，程序中每个定义图纸的地方都要如此写
# fig = plt.figure(1)
# ax1 = fig.add_subplot(111, frameon=False)
# ax1.set_xticks([])
# ax1.set_yticks([])