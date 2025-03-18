# 引入opencv
import cv2
# 引入numpy数组
import numpy as np
from torch.utils.tensorboard.summary import video

from panelDetection import img2hls, transFormBinary, open_Close_Cal, getLRfits, drawPanel, cal_radius

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# 定义一个方法用于检测车道
def pipPanel(img):
    # 将传入的图片转换为HLS空间  便于后续进行颜色的分离
    color_binary = img2hls(img)
    #     获取转换以后图片的尺寸
    img_shape = color_binary.shape

    #     定义图像偏移时的变化量
    offset_x = 160
    offset_y = 0

    #     定义两组点 对图像进行透视变化
    pts1 = np.float32([[img_shape[1] * 0.4, img_shape[0] * 0.7],
                       [img_shape[1] * 0.6, img_shape[0] * 0.7],
                       [img_shape[1] * 1 / 8, img_shape[0]],
                       [img_shape[1] * 7 / 8, img_shape[0]]])
    #     定义两个新的坐标点
    pts2 = np.float32([[offset_x, offset_y],
                       [img_shape[1] - offset_x, offset_y],
                       [offset_x, img_shape[0] - offset_y],
                       [img_shape[1] - offset_x, img_shape[0] - offset_y]])
    #     在图像上绘制一个黑色的矩形框 屏蔽图像中间部分 减少非车道区域的干扰
    cv2.rectangle(color_binary, [int(img_shape[1] * 0.4 + 20), int(img_shape[0] * 0.7)],
                  [int(img_shape[1] * 0.6 - 20), int(img_shape[0])],
                  color=(0, 0, 0), thickness=cv2.FILLED)
    # 对图像进行透视矫正
    correct_img = transFormBinary(color_binary, pts1, pts2)

    # 对变化以后的图像进行膨胀和腐蚀操作  目的时：消除噪声和连接断开的车道线
    eroded_img = open_Close_Cal(correct_img, 10, 5)
    # 从处理后的图像中提取左右车道的拟合参数
    left_fit, right_fit = getLRfits(eroded_img)
    # 在图像上绘制车道线
    out_img = drawPanel(eroded_img, left_fit, right_fit)

    # 将车道的视觉转换为原来的视角
    out_img = transFormBinary(out_img, pts2, pts1)
    # 将原图像和绘制了车道线的图像进行按位或操作，将刚刚识别处理的车道线叠加在原图上
    out_img = cv2.bitwise_or(img, out_img)
    # 计算车道线的曲率半径
    cal_radius(out_img, left_fit, right_fit)

    cv2.imshow("color_binary", out_img)


#


if __name__ == '__main__':
    video = cv2.VideoCapture('./img/img/up.mp4')
    #     获取加载成功视频的帧数
    fps = video.get(cv2.CAP_PROP_FPS)
    print("fps", fps)
    # 读取视频流
    success, frame = video.read()
    # 显示加载的视频
    cv2.imshow("frame", frame)
    while success:
        # 将刚刚加载的视频流中的图片导入
        pipPanel(frame)
        # 不断的在视频中读取图片帧数
        success, frame = video.read()
        # 每次延时1ms  如果按下了按键q  则退出该循环
        if cv2.waitKey(1) == ord("q"):
            break
        # 计算每一帧所需要的时间
        cv2.waitKey(int(1000 / fps))
    #     释放摄像头资源
    video.release()
