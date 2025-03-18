import cv2  # 导入opencv
import numpy as np  # numpy 作为  np

# ./img/img/up01.png  不符合python的数据类型
img = cv2.imread("./img/img/up01.png")  # 写入要图片的路径以及文件名
hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
channels_h = hls[:, :, 0]
channels_l = hls[:, :, 1]
channels_s = hls[:, :, 2]
# sobel算子提取边缘
sobel_x = cv2.Sobel(channels_l, -1, 1, 0)
sobel_x_2 = cv2.Sobel(channels_l, -1, 1, 0, borderType=cv2.BORDER_REPLICATE)  # sobel_x 变量将存储所有检测的边缘信息,
abs_sobel_x = np.absolute(sobel_x)
abs_sobel_x_2 = np.absolute(sobel_x_2)
# 归一化sobel结果
scaled_sobel = np.uint8(abs_sobel_x / np.max(abs_sobel_x) * 255)

"""
# 使用比较高效的布尔索引进行二值化
# 将170作为下限阈值,255为上限阈值,基于车道线在图像中的亮度特征的分析以及多次实验
# 同时满足"<170"和">255"条件的为True
"""
# 归一化结果进一步二值化
sx_binary = np.zeros_like(scaled_sobel)
sx_binary[(170 <= scaled_sobel) & (scaled_sobel <= 255)] = 255

s_binary = np.zeros_like(channels_s)
s_binary[(100 <= channels_s) & (channels_s <= 255)] = 255
# 饱和度二值化
color_binary = (sx_binary | s_binary)
cv2.imshow("color", color_binary)
cv2.waitKey(0)

cv2.imwrite("panel11.png", color_binary)
