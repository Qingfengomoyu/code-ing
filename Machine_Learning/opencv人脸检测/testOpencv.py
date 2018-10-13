import cv2
import os
from numpy import *

def getFileName(filepath):
    fileName=[]
    fileList=os.listdir(filepath)
    m=len(fileList)
    for i in range(m):
        fileNameStr=fileList[i]
        fileName.append(os.path.join(filepath,fileNameStr))
    return fileName


fileNameList=getFileName("G:\docu/3D-Face-BMP1-30\BMP1-30/001")
# filepath = "G:\docu/3D-Face-BMP1-30\BMP1-30/001/001-001.bmp"
for name in fileNameList:
    print(name)
    img = cv2.imread(name) # 读取图片
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 转换灰色
    # OpenCV人脸识别分类器
    classifier = cv2.CascadeClassifier( "C:/Users/fuyu\Downloads/haarcascade_frontalface_default.xml" )
    color = (0, 255, 0) # 定义绘制颜色
    # 调用识别人脸
    faceRects = classifier.detectMultiScale( gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
    if len(faceRects): # 大于0则检测到人脸
        for faceRect in faceRects: # 单独框出每一张人脸
            x, y, w, h = faceRect
            # 框出人脸
            cv2.rectangle(img, (x, y), (x + h, y + w), color, 2)
            # 左眼
            cv2.circle(img, (x + w // 4, y + h // 4 + 30), min(w // 8, h // 8), color)
            #右眼
            cv2.circle(img, (x + 3 * w // 4, y + h // 4 + 30), min(w // 8, h // 8), color)
            #嘴巴
            cv2.rectangle(img, (x + 3 * w // 8, y + 3 * h // 4), (x + 5 * w // 8, y + 7 * h // 8), color)
    cv2.startWindowThread()  # 加在这个位置
    '''
    1、 cv2.imread()：读入图片，共两个参数，第一个参数为要读入的图片文件名，第二个参数为如何读取图片，包括cv2.IMREAD_COLOR：读入一副彩色图片；
    cv2.IMREAD_GRAYSCALE：以灰度模式读入图片；
    cv2.IMREAD_UNCHANGED：读入一幅图片，并包括其alpha通道。
    2、cv2.imshow()：创建一个窗口显示图片，共两个参数，第一个参数表示窗口名字，可以创建多个窗口中，但是每个窗口不能重名；第二个参数是读入的图片。
    3、cv2.waitKey()：键盘绑定函数，共一个参数，表示等待毫秒数，将等待特定的几毫秒，看键盘是否有输入，返回值为ASCII值。
    如果其参数为0，则表示无限期的等待键盘输入。
    4、cv2.destroyAllWindows()：删除建立的全部窗口。
    5、cv2.destroyWindows()：删除指定的窗口。
    6、cv2.imwrite()：保存图片，共两个参数，第一个为保存文件名，第二个为读入图片。'''
    cv2.imshow("image", img) # 显示图像
    # c = cv2.waitKey(10)
    cv2.waitKey(500)
    cv2.destroyAllWindows()
