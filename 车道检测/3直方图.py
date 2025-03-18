import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 读取图像
# 原图片
img = cv2.imread("./img/img/up01.png")
# 处理后的图片
correct_image = cv2.imread("./panel2.png")

# 灰度化
correct_image = cv2.cvtColor(correct_image, cv2.COLOR_BGR2GRAY)
img_shape = correct_image.shape
print(pd.DataFrame(correct_image))

"""——————————————————————————————————————————开运算与闭运算————————————————————————————————————————————
开运算:先腐蚀后膨胀,用于去除图形中的小噪点、孤立的小点,腐蚀多余的像素点

原理:
    腐蚀操作阶段,使用一个结构元素(矩形、圆形、其他形状)逐个滑动。
    当遇到不符合物体(通常是白色)状态时,就会腐蚀掉(背景像素、通常时黑色的)
    就可以去除小的噪点、和微小的物体。
    膨胀阶段阶段：
                使用一个结构元素(矩形、圆形、其他形状)逐个滑动。
    当遇到有一个像素是目标像素时,就会膨胀其结构元素中心点的像素
    不会回复之前已经腐蚀的像素点

应用:
    图像去噪、物体分离
    
    
    
    
闭运算:先膨胀后腐蚀。填充图像中的小孔、小裂缝。膨胀特点的元素.

原理：
     膨胀阶段：
                         使用一个结构元素(矩形、圆形、其他形状)逐个滑动。
    当遇到有一个像素是目标像素时,就会膨胀其结构元素中心点的像素
    不会回复之前已经腐蚀的像素点
     
     腐蚀阶段：
     ,使用一个结构元素(矩形、圆形、其他形状)逐个滑动。
    当遇到不符合物体(通常是白色)状态时,就会腐蚀掉中心像素点(背景像素、通常时黑色的)
    就可以去除小的噪点、和微小的物体。避免过度膨胀。
    
适用场景:
    图像修复
    物体轮廓修复
    
    
        
    腐蚀
    定义:形态学操作,可以是图像中目标物体(白的或者比较亮的物体)经过一定的收缩，
    在二进制图像中,(只有 0   黑色       1    白色)
    会将目标物体(白色区域)的边界像素根据一定的规则来变为背景颜色(黑色)
    
    原理:
            结构元素的小矩阵，进行滑动,,对于每个像素位置,,当结构元素所覆盖的像素与图像中心点的元素的形状不完全匹配时，就要进行覆盖
    
    
    
    膨胀
    
    
"""
"""
1、创建一个结构元素(如一个3*3的矩形)
"""
# MORPH_RECT 表示创建矩形结构元素
# MORPH_ELLIPSE 表示创建圆形结构元素
# MORPH_CROSS 表示创建十字结构元素

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
print(f"kernel is:{kernel}")
dilate_image = cv2.dilate(correct_image, kernel)
for i in range(1000):  # 开闭运算次数

    # 腐蚀
    # erode 使用定义好的元素结构kernel对图像correct_image进行腐蚀
    erode_image = cv2.erode(dilate_image, kernel)

    # 腐蚀
    # erode 使用定义好的元素结构kernel对图像correct_image进行腐蚀
    erode_image = cv2.erode(erode_image, kernel)
    # 显示经过一次闭运算之后的腐蚀图像

    # 膨胀
    # dilate 使用定义好的元素结构对图像correct_image进行膨胀
    # 膨胀操作会让图像中白色(较亮)区域扩大
    dilate_image = cv2.dilate(erode_image, kernel)
    # 膨胀
    # dilate 使用定义好的元素结构对图像correct_image进行膨胀
    # 膨胀操作会让图像中白色(较亮)区域扩大
    dilate_image = cv2.dilate(dilate_image, kernel)

# 腐蚀
# erode 使用定义好的元素结构kernel对图像correct_image进行腐蚀
erode_image = cv2.erode(dilate_image, kernel)

cv2.imshow("erode_image", erode_image)
cv2.waitKey(0)
# 直方图
# np.sum(...,axis=0) 表示沿着列方向,将图像中矩阵的数值加到一个新的矩阵中
histogram = np.sum(dilate_image[:, :], axis=0)  # 类似于横向压缩为一个矩阵？
print(pd.DataFrame(histogram))
# 使用matplotlib绘制直方图
# 横坐标：
# 是图像的列索引(范围从0开始 到图像的长度，通过np.arrange(0,len(histogram))获得,(例子结果是1280)
# 纵坐标：
# 对应的列像素总和,histogram
# r red g green blue y yellow
# - 虚线 .-点虚线 ...................

plt.plot(np.arange(0, len(histogram)), histogram, 'r-')
plt.show()
# 猜测：这个直方图揭示了当纵坐标为多少时横坐标是非0的
