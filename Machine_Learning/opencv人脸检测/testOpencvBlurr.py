import cv2
import numpy as np
img_path=r'C:\Program Files\Python36\project\bug.png'
# 读取图片
img=cv2.imread(img_path)
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#bgr转化为灰度图
# 颜色空间转换，大多数看到的彩色图片都是RGB类型，但是在进行图像处理时，需要用到灰度图、二值图、HSV、HSI等颜色制式
blurred=cv2.GaussianBlur(gray,(9,9),0)
gradx=cv2.Sobel(blurred,ddepth=cv2.CV_32F,dx=1,dy=0)
grady=cv2.Sobel(blurred,ddepth=cv2.CV_32F,dx=0,dy=1)
gradient=cv2.subtract(gradx,grady)
gradient=cv2.convertScaleAbs(gradient)
blurred=cv2.GaussianBlur(gradient,(9,9),0)
(_,thresh)=cv2.threshold(blurred,90,255,cv2.THRESH_BINARY)

kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(25,25))
closed=cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel)
closed=cv2.erode(closed,None,iterations=4)
closed=cv2.dilate(closed,None,iterations=4)

(_,cnts,_)=cv2.findContours(closed.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
c=sorted(cnts,key=cv2.contourArea,reverse=True)[0]
rect=cv2.minAreaRect(c)
box=np.int0(cv2.boxPoints(rect))

draw_img=cv2.drawContours(img.copy(),[box],-1,(0,0,255),3)

xs=[i[0] for i in box]
ys=[i[1] for i in box]
x1=min(xs)
x2=max(xs)
y1=min(ys)
y2=max(ys)
hight=y2-y1
width=x2-x1
crop_img=img[y1:y1+hight,x1:x1+width]
cv2.imshow('img',img)
cv2.imshow('blurred',blurred)
cv2.imshow('gray',gray)
cv2.imshow('gradx',gradx)
cv2.imshow('grady',grady)
cv2.imshow('final',gradient)
cv2.imshow('thresh',thresh)
cv2.imshow('closed',closed)
cv2.imshow('draw_img',draw_img)
cv2.imshow('crop_img',crop_img)
# cv2.imwrite(filename,img,params=[])
# 第三个参数针对特定的格式，对于jpeg，表示图像的质量，用0-100表示，默认95，cv2.IMWRITE_JPEG_QUALITY类型为long，必须转换为int
#[int（cv2.IMWRITE_JPEG_QUALITY,5）]
# 对于png，第三个参数表示压缩级别，cv2.IMWRITE_PNG_COMPRESSION,从0到9，压缩级别越高，图像尺寸越小，默认级别为3，
# [int（cv2.IMWRITE_PNG_COMPRESSION），5]
cv2.waitKey(20180714)
# 等待键盘时间，如果设为0，进入死循环




# import cv2
# import numpy as np
#
#
# def get_image(path):
#     #获取图片
#     img=cv2.imread(path)
#     gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#
#     return img, gray
#
# def Gaussian_Blur(gray):
#     # 高斯去噪
#     blurred = cv2.GaussianBlur(gray, (9, 9),0)
#
#     return blurred
#
# def Sobel_gradient(blurred):
#     # 索比尔算子来计算x、y方向梯度
#     gradX = cv2.Sobel(blurred, ddepth=cv2.CV_32F, dx=1, dy=0)
#     gradY = cv2.Sobel(blurred, ddepth=cv2.CV_32F, dx=0, dy=1)
#
#     gradient = cv2.subtract(gradX, gradY)
#     gradient = cv2.convertScaleAbs(gradient)
#
#     return gradX, gradY, gradient
#
# def Thresh_and_blur(gradient):
#
#     blurred = cv2.GaussianBlur(gradient, (9, 9),0)
#     (_, thresh) = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY)
#
#     return thresh
#
# def image_morphology(thresh):
#     # 建立一个椭圆核函数
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
#     # 执行图像形态学, 细节直接查文档，很简单
#     closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
#     closed = cv2.erode(closed, None, iterations=4)
#     closed = cv2.dilate(closed, None, iterations=4)
#
#     return closed
#
# def findcnts_and_box_point(closed):
#     # 这里opencv3返回的是三个参数
#     (_, cnts, _) = cv2.findContours(closed.copy(),
#         cv2.RETR_LIST,
#         cv2.CHAIN_APPROX_SIMPLE)
#     c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
#     # compute the rotated bounding box of the largest contour
#     rect = cv2.minAreaRect(c)
#     box = np.int0(cv2.boxPoints(rect))
#
#     return box
#
# def drawcnts_and_cut(original_img, box):
#     # 因为这个函数有极强的破坏性，所有需要在img.copy()上画
#     # draw a bounding box arounded the detected barcode and display the image
#     draw_img = cv2.drawContours(original_img.copy(), [box], -1, (0, 0, 255), 3)
#
#     Xs = [i[0] for i in box]
#     Ys = [i[1] for i in box]
#     x1 = min(Xs)
#     x2 = max(Xs)
#     y1 = min(Ys)
#     y2 = max(Ys)
#     hight = y2 - y1
#     width = x2 - x1
#     crop_img = original_img[y1:y1+hight, x1:x1+width]
#
#     return draw_img, crop_img
#
# def walk():
#
#     img_path = r'C:\Program Files\Python36\project\bug.png'
#     # save_path = r'C:\Users\aixin\Desktop\chongzi_save.png'
#     original_img, gray = get_image(img_path)
#     blurred = Gaussian_Blur(gray)
#     gradX, gradY, gradient = Sobel_gradient(blurred)
#     thresh = Thresh_and_blur(gradient)
#     closed = image_morphology(thresh)
#     box = findcnts_and_box_point(closed)
#     draw_img, crop_img = drawcnts_and_cut(original_img,box)
#
#     # 暴力一点，把它们都显示出来看看
#
#     cv2.imshow('original_img', original_img)
#     cv2.imshow('blurred', blurred)
#     cv2.imshow('gradX', gradX)
#     cv2.imshow('gradY', gradY)
#     cv2.imshow('final', gradient)
#     cv2.imshow('thresh', thresh)
#     cv2.imshow('closed', closed)
#     cv2.imshow('draw_img', draw_img)
#     cv2.imshow('crop_img', crop_img)
#     cv2.waitKey(20171219)
#     # cv2.imwrite(save_path, crop_img)
#
# walk()