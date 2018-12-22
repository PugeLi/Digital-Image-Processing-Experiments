import sys

import cv2
import numpy as np


def region_grow(img, edge, seedx, seedy, threshold, flag):
    """
    region grow算法生成背景mask
    :param img: - 源图片
    :param edge: - canny算子提取的边缘
    :param seedx: - 种子点的x值
    :param seedy: - 种子点的y值
    :param threshold: - 距离阈值
    :param flag: - True表示使用4邻域， False表示使用8邻域
    :return: 生成的背景掩码，黑色为前景部分，白色为背景部分
    """
    h, w, _ = np.shape(img)
    img = img.copy().astype(np.double)
    mask = np.zeros((h, w))
    is_visited = np.zeros((h, w)).astype(np.uint8)
    cur = 255
    mask[seedx, seedy] = cur
    is_visited[seedx, seedy] = 1
    queue = [(seedx, seedy)]

    while len(queue) > 0:
        x, y = queue.pop()
        pixel = img[x, y, :]
        for yy in range(-1, 2):
            for xx in range(-1, 2):
                if flag and abs(yy) and abs(xx):
                    continue
                cx = x + xx
                cy = y + yy
                if cx >= 0 and cx < h and cy >= 0 and cy < w:
                    if np.linalg.norm(img[cx, cy, :] - pixel) < threshold \
                            and is_visited[cx, cy] == 0 and edge[cx, cy] != 255:
                        queue.append((cx, cy))
                        mask[cx, cy] = cur
                        is_visited[cx, cy] = 1
    return mask.astype(np.uint8)


# 获取图片点击的区域
def get_coordinate(event, x, y, flags, param):
    global ix, iy
    if event == cv2.EVENT_LBUTTONDOWN:
        ix, iy = x, y
        cv2.destroyAllWindows()

if __name__ == '__main__':

    threshold = 45 # 距离阈值，根据图片调整
    canny_low = 140  # canny算子的下阈值，根据图片调整
    canny_high = 280 # canny算子的上阈值，根据图片调整

    img_path = './img/img.png'

    if len(sys.argv) < 2:
        print('Param Error: python remove_bg.py /path/to/img')
        exit(0)
    else:
        img_path = sys.argv[1]

    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(gray, canny_low, canny_high)

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('image', get_coordinate)
    cv2.imshow('image', img)
    cv2.waitKey()

    print('运行中....')

    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    mark = region_grow(hls, edge, iy, ix, threshold, True)

    # 对于mask先膨胀(对应于腐蚀黑点)，后腐蚀，去除孤立区域
    mark = cv2.dilate(mark, None, iterations=15) # iterations可变
    mark = cv2.erode(mark, None, iterations=15)
    h, w, _ = np.shape(img)

    # 去除标中的背景，置为白色
    white = np.array([255, 255, 255])
    for i in range(h):
        for j in range(w):
            if mark[i, j] == 255:
                img[i, j] = white
    cv2.imwrite('result.png', img)
    print('算法结束，结果存在result.png')
