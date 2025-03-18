import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

correct_image = cv2.imread('./panel2.png')

# 灰度处理
correct_image = cv2.cvtColor(correct_image, cv2.COLOR_BGR2GRAY)

# 创建矩形结构元素。
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
"""
腐蚀操作就是使用核在原图（二值化图）上进行从左到右、从上到下的滑动（也就是从图像的左上角开始，滑动到图像的右下角）。
在滑动过程中，令核值为1的区域与被核覆盖的对应区域进行相乘，得到其最小值，该最小值就是卷积核覆盖区域的中心像素点的新像素值
，接着继续滑动。由于操作图像为二值图，所以不是黑就是白，这就意味着，在被核值为1覆盖的区域内，只要有黑色（像素值为0），那么
该区域的中心像素点必定为黑色（0）。这样做的结果就是会将二值化图像中的白色部分尽可能的压缩，如下图所示，该图经过腐蚀之后，“
变瘦”了。
膨胀相反
"""


"""
1.膨胀，物体边界扩展，可以链接断开的部分
2.对膨胀的图像进行腐蚀消除噪点
3.再次腐蚀，进一步细化图像（消除不需要的部分）
4.再次膨胀，相互配合，调整图像形态
5.又一次膨胀
6.又一次腐蚀
具体以实际情况为主
"""
# 膨胀
dilate_image = cv2.dilate(correct_image, kernel)
# 腐蚀
erode_image = cv2.erode(dilate_image, kernel)

# 腐蚀
erode_image = cv2.erode(erode_image, kernel)
# 膨胀
dilate_image = cv2.dilate(erode_image, kernel)
# 膨胀
dilate_image = cv2.dilate(dilate_image, kernel)
# 腐蚀
erode_image = cv2.erode(dilate_image, kernel)
# 膨胀
dilate_image = cv2.dilate(erode_image, kernel)
# 腐蚀
erode_image = cv2.erode(dilate_image, kernel)
# cv2.imshow("erode_image", erode_image)
# cv2.waitKey(0)


"""
一、获取车道线的初始位置
"""

"1.1 找到两条车道线"
# 直方图图像会有两个凸点，代表了车道线，因为数组的其他地方为0，有车道线的地方才有数字，每列相加为一维数组后只有有车道线的那一块会有数值。
histogram = np.sum(dilate_image[:, :], axis=0)  # 类似于纵向压缩为一个矩阵
plt.plot(np.arange(0, len(histogram)), histogram, 'r-')
plt.show()
"1.2 计算两条车道线的中心线并以此划分左右车道的搜索范围"
# 计算直方图(每列像素值相加数组)中的中点位置，用于分割划分左右车道的搜索范围
midpoint = np.array(histogram.shape[0] / 2, dtype=np.int32)

# 在直方图的左半部分,寻找像素点累加的最大位置，这个位置作为左车道线初始搜索的大致横坐标的起点。
left_x_base = np.argmax(histogram[:midpoint])
# 在直方图的右半部分,寻找像素点累加的最大位置+中点位置的偏移量，这个位置作为右车道线初始搜索的大致横坐标的终点。
right_x_base = np.argmax(histogram[midpoint:]) + midpoint  # histogram[midpoint:]只是中点到右车道线的距离

# 根据上面的计算初始化车道线
# 车道检测当前位置，初始化左车道线的当前横坐标
left_x_current = left_x_base
right_x_current = right_x_base

"""
二、创建滑动窗口用于检测车道线，并为正式滑动作准备
"""

"2.1 设置滑动窗口的数量、高度、检测的水平范围、最小像素点阈值"
# 设置滑动窗口的数量，数量可以决定在图像的垂直方向划分多个区域来搜索车道线
m_windows = 9
# 计算每个滑动窗口的高度通过图像的总高度/窗口数量
window_height = int(erode_image.shape[0] / m_windows)  # 之所以要转换为int,是因为浮点数通常只有两位数不够精确，并且影响运行速度。
# 设置x的检测范围，这里是滑动窗口宽度的一半，手动指定一个值_____，可以确定每个滑动窗口内左右车道线可能出现的水平范围，
margin = 100
# 6.设置最小像素点阈值，用于统计每个滑动区域的非0像素个数，当窗口内的非0像素个数小于阈值是，就说明可能不是车道线，对中心点位置进行更新
minpix = 50

"2.2 找到整个图像中不为零的像素点的坐标"
# 获取图像中像素值不为0的坐标,nonzero()返回值是两个数组，非零点的纵坐标(行)，非0点的纵坐标(行索引)和横坐标(列索引)
non_zero = erode_image.nonzero()
print(f"________________________________________________________________________________{non_zero}")
# 将此数组的纵、坐标提取出来，并且进行类型转换为numpy
non_zero_y = np.array(non_zero[0])  # 纵坐标
non_zero_x = np.array(non_zero[1])  # 横坐标

"2.3 初始化两个索引，分别用于记录那些在左、右车道线搜索中找到的非0数值"
# 用于记录搜索窗口的左右车道线的非0数值在nonzero_y和x的索引。初始化为空。
left_lane_inds = []
right_lane_inds = []

"""
三、 开始用滑动窗口搜索车道线,遍历该图中的每一个窗口，从底部窗口开始向上遍历
"""
# m_windows是滑动窗口个数
for window in range(m_windows):
    "3.1 窗口纵坐标范围"
    # 设置窗口的y的检测范围(纵坐标范围)
    win_y_low = erode_image.shape[0] - (window + 1) * window_height
    win_y_high = erode_image.shape[0] - window * window_height

    "3.2 窗口横坐标范围"
    # 左车道线x的范围，根据当前左车道的横坐标位置与设置margin(检测范围100)来确定当前车道线坑出现的水平范围
    win_x_left_low = left_x_current - margin
    win_x_left_high = left_x_current + margin
    # 右车道线x的范围，根据当前右车道的横坐标位置与设置margin(检测范围100)来确定当前车道线坑出现的水平范围
    win_x_right_low = right_x_current - margin
    win_x_right_high = right_x_current + margin

    "3.3 收集在当前滑动窗口中的非零像素点的索引"
    # good_left_inds和good_right_inds是一个数组，它包含了在当前滑动窗口中被识别为属于左右车道线的非零像素点的索引。这些索引指向non_zero_x和non_zero_y数组中的位置，即它们对应于图像中非零像素点的横坐标
    good_left_inds = ((non_zero_y >= win_y_low) & (non_zero_y < win_y_high) &  # 表示处于窗口纵坐标范围内的非0值的索引
                      (non_zero_x >= win_x_left_low) & (non_zero_x < win_x_left_high)).nonzero()  # 表示处于窗口左车道横坐标范围的非0值的索引
    good_right_inds = ((non_zero_y >= win_y_low) & (non_zero_y < win_y_high) &  # 表示处于窗口纵坐标范围内的非0值的索引
                       (non_zero_x >= win_x_right_low) & (non_zero_x < win_x_right_high)).nonzero()  # 表示处于窗口右车道横坐标范围的非0值的索引

    "3.4 将收集到的索引添加到分为左右车道分别添加到列表中"
    # 将在车道线搜索窗口内的非0点的索引添加到记录在车道索引的列表中
    left_lane_inds.append(good_left_inds)
    right_lane_inds.append(good_right_inds)

    "3.5 通过阈值检验车道是否在窗口内，如果不在就更新横坐标(初始横坐标是根据上面的一维纵坐标相加矩阵)"
    # 不直接用non_zero_x和non_zero_y原因就是good_left_inds和good_right_inds是经过窗口的阈值(minpix)检验的，更能说明是车道线，而non_zero_x和non_zero_y包含大量非车道线数值
    # 如果获取的左车道线搜索窗口内的个数小于最小个数(minpix),则利用这些点的横坐标平均值来进行更新滑动窗口在x轴的车道线
    if len(good_left_inds) > minpix:
        left_x_current = np.mean(non_zero_x[good_left_inds]).astype(dtype=np.int32)  # non_zero 为横坐标索引
    # 如果获取的右车道线搜索窗口内的个数小于最小个数(minpix),则利用这些点的横坐标平均值来进行更新滑动窗口在x轴的车道线
    if len(good_right_inds) > minpix:
        right_x_current = np.mean(non_zero_x[good_right_inds]).astype(dtype=np.int32)  # non_zero 为横坐标索引

"""
四、循环结束，提取获得到的车道线的索引
"""
# 将检测处左右车道点的索引列表合成一个numpy数组，为了统一处理,  axis=1————>按列方向
"4.1 合并，并转为numpy数组"
left_lane_inds = np.concatenate(left_lane_inds, axis=1)
right_lane_inds = np.concatenate(right_lane_inds, axis=1)

"4.2 获取左右车道的横纵坐标"
left_x = non_zero_x[left_lane_inds]
left_y = non_zero_y[left_lane_inds]

right_x = non_zero_x[right_lane_inds]
right_y = non_zero_y[right_lane_inds]
"""------------------------------------------------程序执行至此已经获取了所有车道线的坐标-------------------------------------------------------------------"""
"""五、曲线拟合"""

# 3.用于曲线拟合检测出的点，二次多项式拟合，返回结果是二次项的系数(a,b,c),拟合车道线检测出的点,拟合x = ay
left_fit = np.polyfit(left_y[0], left_x[0], 2)
right_fit = np.polyfit(right_y[0], right_x[0], 2)

"""六、进行车道线可视化"""
"6.1 获取图像行数"
y_max = erode_image.shape[0]

"6，2 创建一个处理后的图像，从灰度图重新转为彩色图"
# np.dstack 是 NumPy 库中的一个函数，用于沿深度方向（第三维）堆叠数组。
# 在这里是从单通道灰度图转为三通道灰度图，三通道意味着虽然本身暂时还没有颜色，但是具备了显示彩色的能力
out_img = np.dstack(
    (erode_image, erode_image, erode_image)) * 255  # *255的原因是：灰度图是二值图像，只有0和1，回到彩色图要变成0-255的范围，白的就是255，黑的仍然是0

"6.3 获得根据车道线进行曲线拟合生成的坐标点"
# 在拟合曲线中获取左、右车道线的像素点通过垂直方向的每一个坐标点y代入拟合的二次多项式公式，进行计算横坐标，从而生成一系列的坐标点
left_points = [[left_fit[0] * y ** 2 + left_fit[1] * y + left_fit[2], y] for y in range(y_max)]
right_points = [[right_fit[0] * y ** 2 + right_fit[1] * y + right_fit[2], y] for y in range(y_max)]
# 左右车道线的像素点进行合并。形成一个总的坐标点。不然可能另一侧坐标点没有画好车就出去了
line_points = np.vstack((right_points, left_points))

"6.4 对绘制的像素点进行处理"
# 对合并后的坐标点进行随机打乱，更加均匀地展示
np.random.shuffle(line_points)  # 线条区分需要更加明显时使用。

"6.5 根据左右车道线的像素位置绘制多边形。效果是看起来像一整个车道？"
# cv2.fillPoly(out_img, [np.array(line_points, dtype=np.int32)], (0, 255, 0))

"6.6 绘制拟合的车道线"
# 遍历每个车道线像素点,在输出图像上以原型绘制
for point in line_points.astype(dtype=np.int32):
    cv2.circle(out_img, point, 10, (0, 255, 0), thickness=5)

"6.7 显示"
# 显示绘制好的车道线的图像
cv2.imshow('Output', out_img)
cv2.waitKey(0)
