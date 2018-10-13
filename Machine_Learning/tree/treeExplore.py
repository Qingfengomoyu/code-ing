from numpy import *
import matplotlib

from tkinter import *
import regressionTrees
matplotlib.use('TkAgg')#设置后端TkAgg
#将TkAgg和matplotlib链接起来
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure



def reDraw(tolS,tolN):
    reDraw.f.clf()#清除画布
    # 添加子图
    reDraw.a=reDraw.f.add_subplot(111)
    if chkBtnVar.get():#检查model tree是否被选中
        if tolN<2:
            tolN=2
        myTree=regressionTrees.createTree(reDraw.rawDat,regressionTrees.modelLeaf,regressionTrees.modelErr,(tolS,tolN))
        yHat=regressionTrees.createForeCast(myTree,reDraw.testDat,regressionTrees.modelTreeEval)
    else:
        myTree=regressionTrees.createTree(reDraw.rawDat,ops=(tolS,tolN))
        yHat=regressionTrees.createForeCast(myTree,reDraw.testDat)
    reDraw.a.scatter( reDraw.rawDat[:,0].tolist(),reDraw.rawDat[:,1].tolist(),s=5)
    reDraw.a.plot(reDraw.testDat,yHat,linewidth=2.0)
    reDraw.canvas.show()

def getInputs():
    '''获取输入框输入的数字'''
    try:#期望输入框输入是整形
        tolN=int(tolNentry.get())
    except:
        tolN=10
        print('enter Interger for tolN')
        tolNentry.delete(0,END)
        tolNentry.insert(0,'10')
    try:#期望输入是浮点型
        tolS=float(tolSentry.get())
    except:
        tolS=1.0
        print('enter float for tolS')
        tolSentry.delete(0,END)
        tolSentry.insert(0,'1.0')
    return tolN,tolS

def drawNewTree():
    tolN,tolS=getInputs()
    reDraw(tolS,tolN)

#创建gui窗口，必须
root=Tk()
reDraw.f=Figure(figsize=(5,4),dpi=100)
reDraw.canvas=FigureCanvasTkAgg(reDraw.f,master=root)
reDraw.canvas.show()
reDraw.canvas.get_tk_widget().grid(row=0,columnspan=3)
# Label是控件，.grid表示按栅格方式放置，参数，row，column是要放置的位置，columnspan代表列跨度，rowspan代表行跨度
# Label(root,text='Plot Place Holder').grid(row=0,columnspan=3)
#
Label(root,text='tolN').grid(row=1,column=0)
# Entry部件是允许单行文本输入的文本框
tolNentry=Entry(root)
tolNentry.grid(row=1,column=1)
# Entry的insert（）方法填充默认显示的内容
tolNentry.insert(0,'10')

Label(root,text='tolS').grid(row=2,column=0)
tolSentry=Entry(root)
tolSentry.grid(row=2,column=1)
tolSentry.insert(0,'1.0')
# Button控件，是按钮，comman参数代表所用的命令
Button(root,text='ReDraw',command=drawNewTree).grid(row=1,column=2,rowspan=3)
# 本来python中int，float，String，Boolean是不可变类型，gui为了方便，就设置了可变类型的IntVar，StringVar，BooleanVar，DoubleVar
chkBtnVar=IntVar()
# Checkbutton是检查按钮是否被按下（松开后）的部件，可点击的框，其状态可以是选定的或未选定的
#
chkBtn=Checkbutton(root,text='model Tree',variable=chkBtnVar)
chkBtn.grid(row=3,column=0,columnspan=2)
# 读取数据
reDraw.rawDat=mat(regressionTrees.loadDataSet('H:\IDM下载\机器学习实战pdf\MLiA_SourceCode\machinelearninginaction\Ch09/sine.txt'))
# # 对第0列数据从小到大按0.1的间隔排成列表，感觉要有list（arange）
reDraw.testDat=arange(min(reDraw.rawDat[:,0]),max(reDraw.rawDat[:,0]),0.01)
reDraw(1.0,10)
root.mainloop()

