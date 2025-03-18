import cv2

if __name__ == '__main__':
    video = cv2.VideoCapture("./img/img/up.mp4")
    # 读取帧率
    fps = video.get(cv2.CAP_PROP_FPS)
    # success代表是否成功读取,frame是视频的第一帧,
    success, frame = video.read()
    # 如果第一帧读取成功
    while success:
        cv2.imshow('frame', frame)
        success, frame = video.read()  # 视频第一帧之后的每一帧，每一次success, frame = video.read()只读一帧,这里循环调用成了每一帧
        # 按键退出
        # 按下空格关闭视频
        if cv2.waitKey(1) == ord(
                ' '):  # cv2.waitKey(1) 函数会等待1毫秒，检查是否有按键被按下。ord(' ') 是一个Python内置函数，它返回字符的ASCII码值。在这个例子中，ord(' ') 返回空格键的ASCII码值，即32
            break
        # 1000ms = 1s   1000/fps  计算每一帧的时间
        cv2.waitKey(int(1000 / int(fps)))  # 防止出现非整数
        print(int(1000 / int(fps)))
    "按下空格退出视频很慢,增加释放资源功能提高程序运行效率"
    # 释放资源
    video.release()
    # 关闭所有窗口
    cv2.destroyAllWindows()

