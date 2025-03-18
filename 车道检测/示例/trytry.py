
import cv2
import numpy as np

def img2hls(img):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    channel_h = hls[:, :, 0]
    channel_l = hls[:, :, 1]
    channel_s = hls[:, :, 2]
    sobel_x = cv2.Sobel(channel_l, -1, 1, 0)
    abs_sobel_x = np.absolute(sobel_x)
    scaled_sobel = np.uint8(abs_sobel_x / np.max(abs_sobel_x) * 255)
    sx_binary = np.zeros_like(scaled_sobel)
    sx_binary[(170 < scaled_sobel) & (scaled_sobel <= 255)] = 255
    s_binary = np.zeros_like(channel_s)
    s_binary[(100 < channel_s) & (channel_s <= 255)] = 255
    color_binary = (sx_binary | s_binary)
    return color_binary

def tansformBinary(color_binary,pts1,pts2):
    img_shape = color_binary.shape
    pts = cv2.getPerspectiveTransform(pts1, pts2)
    correct_image = cv2.warpPerspective(color_binary, pts, (img_shape[1], img_shape[0]))
    return correct_image

def dilated_eroded_img(correct_image):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated_image = cv2.dilate(correct_image, kernel)
    eroded_image = cv2.erode(dilated_image, kernel)
    eroded_image = cv2.erode(eroded_image, kernel)
    dilated_image = cv2.dilate(eroded_image, kernel)
    dilated_image = cv2.dilate(dilated_image, kernel)
    eroded_image = cv2.erode(dilated_image, kernel)
    return eroded_image

def getLRfits(eroded_image):
    histogram = np.sum(eroded_image[:, :], axis=0)
    midpoint = np.array(histogram.shape[0] / 2, dtype=np.int32)
    left_x_base = np.argmax(histogram[:midpoint])
    right_x_base = np.argmax(histogram[midpoint:]) + midpoint

    m_windows = 9
    window_height = int(eroded_image.shape[0] / m_windows)
    non_zero = eroded_image.nonzero()
    non_zero_y = np.array(non_zero[0])
    non_zero_x = np.array(non_zero[1])

    left_x_current = left_x_base
    right_x_current = right_x_base

    margin = 100
    minpix = 50

    left_lane_inds = []
    right_lane_inds = []

    for window in range(m_windows):
        win_y_low = eroded_image.shape[0] - (window + 1) * window_height
        win_y_high = eroded_image.shape[0] - window * window_height
        win_x_left_low = left_x_current - margin
        win_x_left_high = left_x_current + margin
        win_x_right_low = right_x_current - margin
        win_x_right_high = right_x_current + margin

        good_left_inds = ((non_zero_y >= win_y_low) & (non_zero_y < win_y_high) &
                          (non_zero_x >= win_x_left_low) & (non_zero_x < win_x_left_high)).nonzero()

        good_right_inds = ((non_zero_y >= win_y_low) & (non_zero_y < win_y_high) &
                           (non_zero_x >= win_x_right_low) & (non_zero_x < win_x_right_high)).nonzero()

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:
            left_x_current = np.mean(non_zero_x[good_left_inds]).astype(dtype=np.int32)
        if len(good_right_inds) > minpix:
            right_x_current = np.mean(non_zero_x[good_right_inds]).astype(dtype=np.int32)

    left_lane_inds = np.concatenate(left_lane_inds, axis=1)
    right_lane_inds = np.concatenate(right_lane_inds, axis=1)

    left_x = non_zero_x[left_lane_inds]
    left_y = non_zero_y[left_lane_inds]
    right_x = non_zero_x[right_lane_inds]
    right_y = non_zero_y[right_lane_inds]

    left_fit = np.polyfit(left_y[0], left_x[0], 1)
    right_fit = np.polyfit(right_y[0], right_x[0], 1)

    return left_fit,right_fit

def drawPanel(eroded_image,left_fit,right_fit):
    y_max = eroded_image.shape[0]
    out_img = np.dstack((eroded_image, eroded_image, eroded_image)) * 255

    # left_points = [[left_fit[0] * y ** 2 + left_fit[1] * y + left_fit[2], y] for y in range(y_max)]
    # right_points = [[right_fit[0] * y ** 2 + right_fit[1] * y + right_fit[2], y] for y in range(y_max)]

    left_points = [[left_fit[0] * y + left_fit[1], y] for y in range(y_max)]
    right_points = [[right_fit[0] * y + right_fit[1], y] for y in range(y_max)]
    # 将左右车道的像素点进行合并
    line_points = np.vstack((right_points, left_points))
    np.random.shuffle(line_points)
    # 根据左右车道线的像素位置绘制多边形
    # cv2.fillPoly(out_img,np.int_([line_points]),(0,255,0))
    for point in line_points.astype(dtype=np.int32):
        cv2.circle(out_img, point, 10, color=(0, 255, 0), thickness=10)
    return out_img

def cal_radius(img,left_fit,right_fit):

    # 图像中像素个数与实际中距离的比率
    # 沿车行进的方向长度大概覆盖了30米，按照中国公路的标准，宽度为3.7米(经验值)
    ym_per_pix = 30 / 720
    xm_per_pix = 3.7 / 640

    # 计算得到曲线上的每个点
    left_y_axis = np.linspace(0,img.shape[0],img.shape[0] - 1)
    # left_x_axis = left_fit[0] * left_y_axis**2 + left_fit[1] * left_y_axis + left_fit[2]
    left_x_axis = left_fit[0] * left_y_axis + left_fit[1]
    right_y_axis = np.linspace(0, img.shape[0], img.shape[0] - 1)
    # right_x_axis = right_fit[0] * right_y_axis ** 2 + right_fit[1] * right_y_axis + right_fit[2]
    right_x_axis = right_fit[0] * right_y_axis + right_fit[1]

    # 获取真实环境中的曲线
    left_fit_cr = np.polyfit(left_y_axis * ym_per_pix,left_x_axis * xm_per_pix,2)
    right_fit_cr = np.polyfit(right_y_axis * ym_per_pix,right_x_axis * xm_per_pix,2)

    # 获得真实环境中的曲率
    left_curverad = ((1 + (2 * left_fit_cr[0] * left_y_axis * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])

    right_curverad = ((1 + (2 * right_fit_cr[0] * right_y_axis * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_cr[0])

    # 在图像上显示曲率
    cv2.putText(img,'Radius of Curvature = {}(m)'.format(np.mean(left_curverad)),(20,50),0,0.8,color=(255,255,255),thickness=2)
    return img
