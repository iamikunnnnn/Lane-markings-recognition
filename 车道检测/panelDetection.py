import numpy as np
import cv2
import matplotlib.pyplot as plt


def img2hls(img):
    """
    功能：将输入的BGR格式的图像转化为HLS颜色空间，并基于亮度和饱和度通道进行一些列处理，并最终生成二值化图像
    :param img: img输入的图像是彩色图(bgr)
    :return: 二值化图像
    """

    # 将输入的图像从bgr转化为hls
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    # 分别提取HLS图像中的三个通道,提取出来的维度作为图像的宽度*高度
    channels_h = hls[:, :, 0]
    channels_l = hls[:, :, 1]
    channels_s = hls[:, :, 2]

    # 使用sobel算子根据亮度提取边缘
    abs_sobel_x = np.absolute(cv2.Sobel(channels_l, -1, 1, 0))
    abs_sobel_x_2 = np.absolute(
        cv2.Sobel(channels_l, -1, 1, 0, borderType=cv2.BORDER_REPLICATE))  # sobel_x 变量将存储所有检测的边缘信息,

    # 亮度归一化
    scaled_sobel = np.uint8(abs_sobel_x / np.max(abs_sobel_x) * 255)

    # 亮度二值化
    sx_binary = np.zeros_like(scaled_sobel)
    sx_binary[
        (170 <= scaled_sobel) & (scaled_sobel <= 255)] = 255  # 先是取出<170和>255的元素,取出的均为布尔值,再根据布尔值为True的填充为255(在灰度图中即为白色)

    # 提高饱和度区域
    s_binary = np.zeros_like(channels_s)
    s_binary[(100 <= channels_s) & (channels_s <= 255)] = 255

    # 合并处理
    color_binary = (sx_binary | s_binary)

    # 显示
    # cv2.imshow("color", color_binary)
    # cv2.waitKey(0)
    # 保存
    # cv2.imwrite("panel12.png", color_binary)
    return color_binary


def transFormBinary(color_binary, pts1, pts2):
    # 汽车进入车道的角度会变化，为了统一使用透视变换使视角统一
    """
    功能：对输入的二值化图像进行透视变换，改变图像视角。
    :param color_binary: 获取输入二值化图像的形状，后续确定透视变换后，输出图像
    :param pts1: 原始图像上的透视变换点(一般情况下是四边形)
    :param pts2: 目标图像上的透视变换点(一般情况下是四边形)
    :return: 矫正后的图像
    """
    # 获取图像形状,用于后续图像处理区域的定位
    img_shape = color_binary.shape
    # 计算透视变换矩阵
    pts = cv2.getPerspectiveTransform(pts1, pts2)
    # 应用透视变换
    correct_image = cv2.warpPerspective(color_binary, pts, (img_shape[1], img_shape[0]))
    # 绘制填充矩形
    return correct_image


def open_Close_Cal(correct_image, times, ksize):
    """
    功能：对输入的二值化图像进行开闭运算,去除噪点并增强轮廓
    :param correct_image: 输入的图像
    :param times: 开闭运算次数,默认会进行一次开闭运算，运算次数为times+1
    :param ksize: 滑动窗口(元素结构)的大小
    :return: 开闭运算后的图像
    """

    # 创建元素结构
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    # 膨胀
    dilate_image = cv2.dilate(correct_image, kernel)
    for i in range(times):  # 开闭运算次数，
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
    # 腐蚀
    erode_image = cv2.erode(erode_image, kernel)
    # 膨胀
    dilate_image = cv2.dilate(erode_image, kernel)
    return dilate_image


import numpy as np


def getLRfits(eroded_image):
    """
    功能：通过滑动窗口算法在处理图像中寻找对应的左右车道的像素点。然后使用二次多项式拟合，找到对应的像素点
    :param eroded_image: 经过形态学处理后的图像
    :return: 多项式拟合系数
    """
    # 计算直方图，用于初始化车道线
    histogram = np.sum(eroded_image, axis=0)
    # 计算直方图的中点位置，用于分割左右车道
    midpoint = histogram.shape[0] // 2
    # 直方图的左半部分(起点到中点)
    left_base_x = np.argmax(histogram[:midpoint])
    # 直方图的右半部分(中点到最右端)
    right_base_x = np.argmax(histogram[midpoint:]) + midpoint

    # 设置滑动窗口数量
    m_windows = 9
    # 计算每个滑动窗口的高度
    window_height = int(eroded_image.shape[0] / m_windows)

    # 获取图像中像素值不为0的点的横纵坐标
    non_zero = eroded_image.nonzero()
    non_zero_y = np.array(non_zero[0])
    non_zero_x = np.array(non_zero[1])

    # 初始化车道线的当前x位置
    left_x_current = left_base_x
    right_x_current = right_base_x

    # 设置滑动窗口的x轴检测范围
    margin = 100
    # 设置最小像素点阈值
    minpix = 50
    # 记录左右车道线的非零点索引
    left_lane_inds = []
    right_lane_inds = []

    # 从下到上遍历每个窗口，搜索车道线像素点
    for window in range(m_windows):
        # 设置窗口的y轴检测范围
        win_y_low = eroded_image.shape[0] - (window + 1) * window_height
        win_y_high = eroded_image.shape[0] - window * window_height

        # 左右车道的x轴范围
        win_x_left_low = left_x_current - margin
        win_x_left_high = left_x_current + margin
        win_x_right_low = right_x_current - margin
        win_x_right_high = right_x_current + margin

        # 找到处于当前窗口范围内的非零点的索引
        good_left_inds = np.where((non_zero_y >= win_y_low) & (non_zero_y < win_y_high) &
                                  (non_zero_x >= win_x_left_low) & (non_zero_x < win_x_left_high))
        good_right_inds = np.where((non_zero_y >= win_y_low) & (non_zero_y < win_y_high) &
                                   (non_zero_x >= win_x_right_low) & (non_zero_x < win_x_right_high))

        # 将找到的索引添加到左右车道线的索引列表中
        left_lane_inds.append(good_left_inds[0])
        right_lane_inds.append(good_right_inds[0])

        # 如果窗口内的左车道点数量大于最小阈值，更新左车道线的当前位置
        if len(good_left_inds[0]) > minpix:
            left_x_current = np.mean(non_zero_x[good_left_inds])
        # 如果窗口内的右车道点数量大于最小阈值，更新右车道线的当前位置
        if len(good_right_inds[0]) > minpix:
            right_x_current = np.mean(non_zero_x[good_right_inds])

    # 将所有窗口的索引合并
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # 获取左右车道的横纵坐标
    left_x = non_zero_x[left_lane_inds]
    left_y = non_zero_y[left_lane_inds]
    right_x = non_zero_x[right_lane_inds]
    right_y = non_zero_y[right_lane_inds]

    # 使用二次多项式拟合左右车道线
    left_fit = np.polyfit(left_y, left_x, 2)
    right_fit = np.polyfit(right_y, right_x, 2)

    return left_fit, right_fit


def drawPanel(eroded_image, left_fit, right_fit):
    """
    功能：根据上个函数得到的多项式系数，在图像中绘制可视化的车道线
    :param eroded_image: 输入图像，二值化图像（车道线的像素为1，背景为0）
    :param left_fit: 左车道线的多项式系数 [a, b, c]
    :param right_fit: 右车道线的多项式系数 [a, b, c]
    :return: 绘制后的图像
    """
    # 获取图像行数
    y_max = eroded_image.shape[0]
    # 获取图像列数
    x_max = eroded_image.shape[1]

    # 将图像转换为可以显示颜色的三通道图像，并且初始化为白色
    out_img = np.dstack((eroded_image, eroded_image, eroded_image)) * 255

    # 根据左右车道线的多项式系数计算图像中每个垂直坐标点对应的左右车道线位置
    left_points = [[left_fit[0] * y ** 2 + left_fit[1] * y + left_fit[2], y] for y in range(y_max)]
    right_points = [[right_fit[0] * y ** 2 + right_fit[1] * y + right_fit[2], y] for y in range(y_max)]

    # 限制x坐标在图像的有效范围内
    left_points = [(min(max(x, 0), x_max - 1), y) for x, y in left_points]
    right_points = [(min(max(x, 0), x_max - 1), y) for x, y in right_points]

    # 将左右车道线像素点合并，形成纵坐标点用于一起绘制
    line_points = np.vstack((right_points, left_points))

    # 对合并后的坐标点进行随机打乱，更加均匀地展示
    np.random.shuffle(line_points)

    # 遍历每一个车道线，在输出图像上绘制出每一个点，模拟车道线效果
    for point in line_points.astype(dtype=np.int32):
        cv2.circle(out_img, tuple(point), 10, color=(255, 0, 0), thickness=10)

    return out_img


def cal_radius(img, left_fit, right_fit):
    """
    功能：根据拟合后的左右车道线的多项式系数，计算车道在真实世界的曲率半径，并在输入图像上显示曲率半径
    :param img:全是图像
    :param left_fit:左车道线拟合系数
    :param right_fit:右车道线拟合系数
    :return:添加曲率半径后的图像
    """
    # 图像中像素个数与实际中距离的比率
    # 沿车行进的方向长度大概覆盖了30米，按照中国公路的标准，宽度为3.7米(经验值)
    ym_per_pix = 30 / 720
    xm_per_pix = 3.7 / 640

    # 计算得到曲线上的每个点
    left_y_axis = np.linspace(0, img.shape[0], img.shape[0] - 1)
    # left_x_axis = left_fit[0] * left_y_axis**2 + left_fit[1] * left_y_axis + left_fit[2]
    left_x_axis = left_fit[0] * left_y_axis + left_fit[1]
    right_y_axis = np.linspace(0, img.shape[0], img.shape[0] - 1)
    # right_x_axis = right_fit[0] * right_y_axis ** 2 + right_fit[1] * right_y_axis + right_fit[2]
    right_x_axis = right_fit[0] * right_y_axis + right_fit[1]

    # 获取真实环境中的曲线
    left_fit_cr = np.polyfit(left_y_axis * ym_per_pix, left_x_axis * xm_per_pix, 2)
    right_fit_cr = np.polyfit(right_y_axis * ym_per_pix, right_x_axis * xm_per_pix, 2)

    # 获得真实环境中的曲率
    left_curverad = ((1 + (
            2 * left_fit_cr[0] * left_y_axis * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])

    right_curverad = ((1 + (
            2 * right_fit_cr[0] * right_y_axis * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])

    # 在图像上显示曲率
    cv2.putText(img, 'Radius of Curvature = {}(m)'.format(np.mean(left_curverad)), (20, 50), 0, 0.8,
                color=(255, 255, 255), thickness=2)
    return img
