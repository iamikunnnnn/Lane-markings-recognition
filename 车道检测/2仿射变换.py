import cv2
import numpy as np

# 原图片
img = cv2.imread("./img/img/up01.png")
# 处理后的图片
color_binary = cv2.imread("./panel11.png")

# 灰度化
color_binary = cv2.cvtColor(color_binary, cv2.COLOR_BGR2GRAY)
img_shape = color_binary.shape
print(img_shape)
"""
----------------------------------透视变换---------------------------------------------------
汽车进入车道的角度会变化，为了统一使用透视变换使视角统一
仿射变换是一种几何变换，它保持了图像中的直线性和平行性。仿射变换包括平移、缩放、旋转和剪切（shear）等操作。
本质上是一种矩阵计算，矩阵本身就是一种空间变换
这些变换可以组合使用，以实现复杂的图像变换。
"""


"""
1、偏移量设置：
offset_x = 160 和 offset_y = 0：这些偏移量用于调整透视变换后目标图像的位置。
偏移量的选择通常基于实验和图像的具体需求，以确保变换后的图像能够正确地表示车道线.
"""
offset_x = 160
offset_y = 0
"""
2、定义原始图像上的透视变换点：
pts1：这些点定义了原始图像上进行透视变换的四个关键坐标点。定义了需要进行变换的区域
坐标是以图像宽度和高度的比例来表示的，数据类型是np.float32类型的数组。这些点的选择基于图像内容和所需的变换效果。
"""

pts1 = np.float32([
    [img_shape[1] * 0.4, img_shape[0] * 0.7],  # 第一个坐标点,横坐标的图像宽度的0.4倍，纵坐标的图像0.7倍，该例结果为512.0和503.99999999999994
    [img_shape[1] * 0.6, img_shape[0] * 0.7],  # 第二个坐标点,横坐标的图像宽度的0.4倍，纵坐标的图像0.7倍
    [img_shape[1] * 1 / 8, img_shape[0]],  # 第三个坐标点,横坐标的图像宽度的1/8倍，纵坐标就是图像高度
    [img_shape[1] * 7 / 8, img_shape[0]],  # 第四个坐标点,横坐标的图像宽度的7/8倍，纵坐标就是图像高度
])
print(f"放射变换的参数：{img_shape[1] * 0.4, img_shape[0] * 0.7}")
"""
3、定义目标图像上的透视变换点：
pts2：这些点定义了透视变换后目标图像上的四个点。这些点的坐标根据图像的宽度、高度和偏移量进行计算。
目标点的选择是为了将原始图像的特定区域映射到目标图像的特定位置，以便于后续处理。
"""
pts2 = np.float32([
    [offset_x, offset_y],  # 第一个点变换后的坐标，按照偏移量来定位
    [img_shape[1] - offset_x, offset_y],  # 横坐标根据图片的宽度和偏移量进行计算，纵坐标按照偏移量设置0
    [offset_x, img_shape[0] - offset_y],  # 横坐标按照偏移量，纵坐标根据图像高度和偏移量确定
    [img_shape[1] - offset_x, img_shape[0] - offset_y],

])

"""
4、计算透视变换矩阵：
使用cv2.getPerspectiveTransform(pts1, pts2)函数根据给定的源点集pts1和目标点集pts2计算透视变换矩阵pts。
这个矩阵用于将原始图像透视变换为目标图像。
这里的矩阵就是进行仿射变换的关键，仿射变换是一种矩阵运算。
"""
pts = cv2.getPerspectiveTransform(pts1, pts2)

"""
5、应用透视变换：
使用cv2.warpPerspective(color_binary, pts, (img_shape[1], img_shape[0]))函数对原始图像color_binary进行透视变换，
得到校正后的图像correct_image。变换后的图像与原始图像的宽高保持一致。
"""
correct_image = cv2.warpPerspective(color_binary, pts, (img_shape[1], img_shape[0]))  # img_shape[0]和img_shape[1]是元组


"""
6、绘制填充矩形：
在correct_image上绘制一个填充矩形，用于标记或处理图像的特定区域。矩形的坐标根据图像的宽度和高度比例计算得出。
"""
cv2.rectangle(correct_image, [int(img_shape[1] * 0.4 + 20), int(img_shape[0] * 0.7)],
              [int(img_shape[1] * 0.6 + 20), int(img_shape[0])],
              color=(0, 0, 0),
              thickness=cv2.FILLED
              )


cv2.imshow("correct_image", correct_image)
cv2.waitKey(0)

# correct_image是一个图片类型，保存的图片命名要跟上后缀用于确定保存格式
# cv2.imwrite("panel2.png", correct_image)
# cv2.waitKey(0)


