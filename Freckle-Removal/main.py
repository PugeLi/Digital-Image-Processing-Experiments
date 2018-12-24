import numpy as np
import cv2
import getopt
import sys, os


def bilateral_filter(src, radius, color_sigma, space_sigma):
    """
    双边滤波算法
    :param src: - 源图片，必须是单通道
    :param radius: - 邻域半径
    :param color_sigma: - 颜色空间滤波器的sigma值，其值越大，图像越模糊
    :param space_sigma: - 坐标空间中滤波器的sigma值
    :return: 返回滤波后的图片
    """
    rows, cols = np.shape(src)
    dest = np.zeros(np.shape(src))
    space_coeff = -0.5 / (space_sigma * space_sigma)
    color_coeff = -0.5 / (color_sigma * color_sigma)
    d = radius * 2 + 1  # 邻域直径
    # 使用边缘补填充图片，方便计算
    img_padding = np.pad(src, radius, 'edge').astype(np.double)

    # 计算距离模板系数
    space_weight = np.zeros((d, d))  # 存放距离模板系数
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            r = np.sqrt(i * i + j * j)
            space_weight[i, j] = np.exp(r * r * space_coeff)

    for i in range(radius, rows + radius):
        for j in range(radius, cols + radius):
            i_low, i_high = i - radius, i + radius + 1
            j_low, j_high = j - radius, j + radius + 1
            region = img_padding[i_low:i_high, j_low:j_high]
            diff = region - img_padding[i, j]  # 计算邻域每个点与中心点的差值
            color_weight = np.exp(color_coeff * (diff * diff))
            weight = space_weight * color_weight

            sum = np.sum(region * weight)
            w_sum = np.sum(weight)
            dest[i - radius, j - radius] = sum / w_sum

    return dest.astype(np.uint8)


def surface_blur(src, radius, threshold):
    """
    表面模糊算法
    :param src: - 源图片，必须是单通道
    :param radius: - 邻域半径
    :param threshold: - 阈值
    :return: 返回处理后的图片
    """
    h, w = np.shape(src)
    dest = np.zeros(np.shape(src))
    img_padding = np.pad(src, radius, 'edge').astype(np.double)
    for i in range(radius, h + radius):
        for j in range(radius, w + radius):
            i_low, i_high = i - radius, i + radius + 1
            j_low, j_high = j - radius, j + radius + 1
            region = img_padding[i_low:i_high, j_low:j_high]
            diff = 1 - 0.4 * (np.abs(region - img_padding[i, j])) / threshold
            diff = np.where(diff > 0, diff, 0)  # 如果diff小于0，则其置0不参与运算
            diff_sum = np.sum(diff)
            if diff_sum == 0.0:  # 分母为0，则使用原来的值
                dest[i - radius, j - radius] = img_padding[i, j]
            else:
                sum = np.sum(diff * region)
                dest[i - radius, j - radius] = sum / diff_sum
    dest = np.where(dest < 255, dest, 255)  # 超过255的部分，置为255
    return dest.astype(np.uint8)


if __name__ == '__main__':
    img_path = None

    # 默认参数部分
    algorithm = 'bf'
    radius = 13
    threshold = 45
    color_sigma = 45
    space_sigma = 100

    try:
        opts, args = getopt.getopt(sys.argv[1:], 'i:a:r:t:c:s:',
                                   ['img_path=', 'algorithm=', 'radius=', 'threshold=', 'color_sigma=', 'space_sigma='])
    except getopt.GetoptError:
        print(
            'Usage: python main.py -i <img_path> [-a <algorithm>] [-r <radius>] [-t <threshold>] [-c <color_sigma>] [-s <space_sigma>]')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-a', '--algorithm'):
            algorithm = arg
        elif opt in ('-i', '--img_path'):
            img_path = arg
        elif opt in ('-r', '--radius'):
            radius = int(arg)
        elif opt in ('-t', '--threshold'):
            threshold = int(arg)
        elif opt in ('-c', '--color_sigma'):
            color_sigma = int(arg)
        elif opt in ('-s', '--space_sigma'):
            space_sigma = int(arg)

    if not img_path or not os.path.exists(img_path):
        print('图片参数有误,请给出正确图片地址')
        exit(0)

    file_name = img_path.split('/')[-1]
    prefix = file_name.split('.')[0]
    img = cv2.imread(img_path)
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    y = yuv[:, :, 0].copy()

    if algorithm == 'bf':
        # 双边滤波处理
        y_new = bilateral_filter(y, radius, color_sigma, space_sigma)
        edge = cv2.Canny(y_new, 30, 80)
        edge_add = y_new + edge * 0.03  # 增强边缘
        yuv[:, :, 0] = edge_add
        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        cv2.imwrite(prefix + '_bilateral_filter.png', bgr)
    elif algorithm == 'sb':
        # 表面模糊处理
        y_new = surface_blur(y, radius, threshold)
        edge = cv2.Canny(y_new, 30, 80)
        edge_add = y_new + edge * 0.03  # 增强边缘
        yuv[:, :, 0] = edge_add
        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        cv2.imwrite(prefix + '_surface_blur.png', bgr)
