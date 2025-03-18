"""
知识点：
1、sobel：边缘检测算子（本质上是卷积，只是卷积后的图像表现了梯度信息）
1.1 卷积核：
Gx： [-1  0  1,-2  0  2,-1  0  1]Gy： [-1  -2  -1,0  0  0,1  2  1]
解答：为什么选择这个卷积核，当出现边缘时，左右或者上下的差值大，-的或者+的一边会大于另一边，绝对值就会大，如果与卷积核进行卷积的地方数值差不多最后结果则接近于0，
有2的原因时因为sobel认为离得近的像素点更重要，即类似于加权

1.1Sobel算子的工作原理：
它使用两个卷积内核，一个用于计算x方向上的梯度，另一个用于计算y方向上的梯度。
每个内核都考虑了像素及其邻域的值，并根据这些值计算梯度。
通过将这两个梯度分量结合起来（通常使用平方和的平方根），我们可以得到梯度的幅值，它表示了图像中每个像素点处的强度变化速率。

1.2 结论：
因此，卷积操作本身并不直接计算梯度，但它可以被用来实现梯度的近似计算。通过设计适当的卷积内核，
我们可以模拟连续导数在离散图像上的行为，从而用于边缘检测和其他图像处理任务。
在Sobel算子等边缘检测算法中，卷积操作是实现这一目的的关键工具。




2、HLS：色相，亮度，饱和度

3、在图像处理领域梯度不是传统意义上的梯度，只是用卷积算出了变化幅度，本质上类似于梯度

4、归一化

5、二值化

6、不同通道融合
"""

import cv2  # 导入opencv
import numpy as np  # numpy 作为  np

# ./img/img/up01.png  不符合python的数据类型
img = cv2.imread("./img/img/up01.png")  # 写入要图片的路径以及文件名

cv2.imshow("1", img)
# cv2.waitKey(0)  # 阻塞以上代码。按下关闭时,才可以关闭

"""
将图像按照HLS方式进行颜色的空间转换
原因:后续提取车道线,HLS颜色空间特性有助于在高亮部分更好的分离车道线(根据亮度以及饱和度的特征来突出车道线部分)

RGB  灰度   三通道转换为单通到

HLS(Hue 色相    Luminance 亮度   Saturation 饱和度)
"""

hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

# 将HLS 三层进行分割，方便后续针对不同通道进行特征处理

# channels_h  表示色相通道 其颜色范围是:0-120为红   120-240  绿色  240 ---360 为蓝

channels_h = hls[:, :, 0]
# channels_l 表示亮度通道，后续利用亮度信息来检测边缘等一些特征
channels_l = hls[:, :, 1]

# channels_l 表示饱和度通道，饱和度信息对于颜色区分鲜艳程度很关键。有助于提取车道线比较明显
channels_s = hls[:, :, 2]

# 展示HLS 三个通道的图像效果，需要查看的话。将对应注释解开
# cv2.imshow("h",channels_h)
#
# cv2.imshow("l", channels_l)
#
# cv2.imshow("s",channels_s)
#
# cv2.waitKey(0)


"""
利用 sobel 计算x 方向的梯度 横向的边缘检测
原因:在于车道本身在图像中呈现纵向延伸的状态,如果使用了纵向检测(沿Y轴检测)，会丢失纵向的边缘信息。不利用准确提取车道
"""
# sobel_x参数：
# 1.src类型numpy.ndarray图像的输入, 必须是灰度图、单通道
# 2.ddepth: 类型int输出图像的深度(数据类型)指定了输出图像的位深度。常用的值-1输出图像与输入图像深度相同
# cv2.cv_8U8位无符号整数cv2.cv_16S16位有符号数cv2.cv_32F32位浮点数cv2.cv_64F64位浮点数选择合适的输出图像深度对于计算结果的精度和表示的范围有影响。
# 3.dx、dy类型int某个方向的阶数, 表示图像在x、y轴的求导次数
# 1表示计算y、x方向的一阶导数0不计算x, y的导数大于1极少用更高阶的导数
# 4.ksize类型intsobel算子的大小, 通常取奇数, 表示计算梯度时所使用的卷积核大小, 常见的大小有3 * 35 * 57 * 7ksize越大, 算子对图像的平滑效果越强, 但是可能会丢失细节
# 5.scale类型float可选的缩放因子, ，默认值1，计算梯度结果比例缩放
# 6.delta类型folat可选的偏移量, m默认值0, 在计算梯度时,.通常用于调整最终图像中的亮度或者对比度
# 7.borderType类型int边缘像素处理方式, 邻域参数决定了怎么处理图像的边界
# cv2.BORDER_CONSTANT使用常数值来填充边界外的像素
# cv2.BORDER_REFLECT边界外的像素值的镜像反射。
# cv2.BORDER_FEPLTCATE边界外的像素值的镜像复制。
# CV2.BORDER_DEFALUT默认方式填充边界:


# sobel输出结果: 函数返回图像在x和y轴的梯度结果, 类型numpy.ndarray, 返回的图像通道是单通道(图像在每一个像素位置的梯度值)


sobel_x = cv2.Sobel(channels_l, -1, 1, 0)
sobel_x_2 = cv2.Sobel(channels_l, -1, 1, 0, borderType=cv2.BORDER_REPLICATE)  # sobel_x 变量将存储所有检测的边缘信息,
# -1表示输出图像的数据类型与输入图像的数据类型保持一致
# 1 表示 在x轴的方向求一阶导数(检测横向边缘)
# 0 在y轴的方向不求导数,(不用纵向检测)
"""
由于sobel计算得到的边缘信息可能存在负数(通过导数计算得到的梯度信息)
而图像像素范围  0-255  的无符号数的8位整数类型
一定要用绝对值的形式来取值，以便后续操作

"""

abs_sobel_x = np.absolute(sobesl_x)
abs_sobel_x_2 = np.absolute(sobel_x_2)
print(abs_sobel_x.shape)
print(abs_sobel_x_2.shape)
# cv2.imshow("sobel1", abs_sobel_x)
# cv2.imshow("sobel2", abs_sobel_x_2)

"""
归一化  0---1之间进行归一化的值全部乘以255
通过归一化到0--255来实现
归一化需要保证颜色正常
"""
# unsigned int size_t
# 将abs_sobel_x中的每个元素除以最大值，得到一个0到1之间的归一化值。再乘上255.
# 作用：1、对低高亮区域进行边缘都增加。2、二值化处理
scaled_sobel = np.uint8(abs_sobel_x / np.max(abs_sobel_x) * 255)

# 展示经过缩放后的sobel结果
# cv2.imshow("scaled_sobel", scaled_sobel)


"二值化原版本"
# """
# 二值化:忽略不重要的地方,单独取出车道线,事实上就是取出轮廓最明显(梯度最大)的
# """
# # 存储之后的二值化处理结果
# # 设定阈值
# # 创建一个与scaled_sobel大小相同的全0数组
# sx_binary = np.zeros_like(scaled_sobel)
# h, w = scaled_sobel.shape
# for i in range(h):
#     for j in range(w):
#         if 170 <= scaled_sobel[i][j] <= 255:
#             sx_binary[i][j] = 255
#
# """
# 示例:大于170且小于等于255这个区域之间的像素作为高亮像素
# """
# # cv2.imshow('sx_sobel',sx_binary)
# # cv2.waitKey(0)
#
# """
# 饱和度处理
# """
#
# s_binary = np.zeros_like(channels_l)
# h, w = channels_s.shape
# for i in range(h):
#     for j in range(w):
#         if 100 <= channels_s[i][j] <= 255:
#             s_binary[i][j] = 255
#
# # cv2.imshow("s_binary", s_binary)
# # cv2.waitKey(0)
#
# """
# 融合高亮区域和饱和度区域,使用逻辑或(|),如果饱和度区域效果很差则不需要进行这一步,
# 逻辑或规则详见numpy优化.py
# """
# color_binary =(sx_binary|s_binary)
# cv2.imshow('color', color_binary)
# cv2.waitKey(0)

"以下是二值化优化版本"
"""
二值化:忽略不重要的地方,单独取出车道线,事实上就是取出轮廓最明显(梯度最大)的

# 使用比较高效的布尔索引进行二值化
# 将170作为下限阈值,255为上限阈值,基于车道线在图像中的亮度特征的分析以及多次实验
# 同时满足"<170"和">255"条件的为True
"""
# 归一化结果进一步二值化
sx_binary = np.zeros_like(scaled_sobel)
sx_binary[(170 <= scaled_sobel) & (scaled_sobel <= 255)] = 255  # 先是取出<170和>255的元素,取出的均为布尔值,再根据布尔值为True的填充为255(在灰度图中即为白色)

# 提高饱和度区域
s_binary = np.zeros_like(channels_s)
s_binary[(100 <= channels_s) & (channels_s <= 255)] = 255

# 合并处理
color_binary = (sx_binary | s_binary)

cv2.imshow("color", color_binary)
cv2.waitKey(0)

cv2.imwrite("panel12.png", color_binary)
