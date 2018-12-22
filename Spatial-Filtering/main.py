import cv2
import numpy as np
import matplotlib.pyplot as plt
import time


class Img:
    def __init__(self, img_dir):
        self.img = cv2.imread(img_dir)

    def adjust_brightness(self, cof=0.3):
        """
        调整图像亮度
        :param cof: - 亮度变化系数，> 1表示增大亮度
        :return: 返回调整完亮度后的图像
        """
        self.copy = cv2.cvtColor(self.img, cv2.COLOR_BGR2YUV)
        self.copy[:, :, 0] = self.copy[:, :, 0] * cof
        self.copy = cv2.cvtColor(self.copy, cv2.COLOR_YUV2BGR)
        return self.copy

    def adjust_saturation(self, cof=0.3):
        """
        调整图像的饱和度
        :param cof: - 饱和度变化系数， > 1表示增大饱和度
        :return: 返回调整完饱和度后的图像
        """
        self.copy = cv2.cvtColor(self.img, cv2.COLOR_BGR2HLS)
        self.copy[:, :, 2] = self.copy[:, :, 2] * cof
        self.copy = cv2.cvtColor(self.copy, cv2.COLOR_HLS2BGR)
        return self.copy

    def adjust_hue(self, cof=0.6):
        """
        调整图像的色度
        :param cof: - 色度变化系数 > 0表示向色谱上角度更大的颜色调整
        :return: 返回调整完色度后的图像
        """
        self.copy = cv2.cvtColor(self.img, cv2.COLOR_BGR2HLS)
        self.copy[:, :, 0] = self.copy[:, :, 0] + 180 * cof
        self.copy = cv2.cvtColor(self.copy, cv2.COLOR_BGR2HLS)
        return self.copy

    def adjust_contrast(self, cof=0.3):
        """
        调整图像的对比度
        :param cof: - 对比度变化系数, > 1 表示增大对比度
        :return: 返回调整完对比度后的图像
        """
        self.copy = cv2.cvtColor(self.img, cv2.COLOR_BGR2YUV)
        mean = np.mean(self.copy[:, :, 0])
        self.copy[:, :, 0] = mean + (self.copy[:, :, 0] - mean) * cof
        self.copy = cv2.cvtColor(self.copy, cv2.COLOR_YUV2BGR)
        return self.copy

    def get_histogram(self):
        """
        统计img自身的直方图，只能为灰度直方图
        :return: 返回统计后的直方图
        """
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        h, w = np.shape(gray)
        hist = np.zeros(256)
        for i in range(h):
            for j in range(w):
                hist[gray[i, j]] += 1
        return hist / np.sum(hist)  # 归一化

    def gray_median_filter(self, gray_img, pad_size=1):
        """
        对于单通道的图像(灰度图)进行中值滤波
        :param gray_img: - 灰度图
        :param pad_size: - 执行滤波前在图像周围补0的维数，同时指示滤波器大小(2*pad_size + 1)
        :return: 返回滤波后的图像，大小与原图一致
        """
        img = np.pad(gray_img.copy(), pad_size, pad_with, padder=0)
        img_filter = img.copy()
        median = int((2 * pad_size + 1) ** 2 / 2)
        h, w = np.shape(img)
        h_low, h_high = pad_size, h - pad_size
        w_low, w_high = pad_size, w - pad_size
        for i in range(h_low, h_high):
            for j in range(w_low, w_high):
                i_low, i_high = i - pad_size, i + pad_size + 1
                j_low, j_high = j - pad_size, j + pad_size + 1
                img_filter[i, j] = np.median(
                    img[i_low:i_high, j_low:j_high].copy().reshape((1, -1))[0])  # [median]
        img_filter = img_filter[h_low:h_high, w_low:w_high]
        return img_filter

    def rgb_median_filter(self, pad_size=1):
        """
        对RGB通道的彩色图像进行中值滤波
        :param pad_size: - 执行滤波前在图像周围补0的维数，同时指示滤波器大小(2*pad_size + 1)
        :return: 返回滤波后的彩色图像以及各个通道分量，大小与原图像一致
        """
        b, g, r = cv2.split(self.img)
        b_filter = self.gray_median_filter(b, pad_size)
        g_filter = self.gray_median_filter(g, pad_size)
        r_filter = self.gray_median_filter(r, pad_size)
        img = cv2.merge((b_filter, g_filter, r_filter))
        return img, b_filter, g_filter, r_filter

    def gray_mean_filter(self, gray_img, filter, pad_size=1):
        """
        对于单通道的图像(灰度图)进行均值滤波
        :param gray_img: - 灰度图
        :param filter: - 滤波器，n维方阵
        :param pad_size: - 执行滤波前在图像周围补0的维数
        :return: 返回滤波后的图像，大小与原图一致
        """
        img = np.pad(gray_img.copy(), pad_size, pad_with, padder=0)
        img_filter = img.copy()
        h, w = np.shape(img)
        h_low, h_high = pad_size, h - pad_size
        w_low, w_high = pad_size, w - pad_size
        for i in range(h_low, h_high):
            for j in range(w_low, w_high):
                i_low, i_high = i - pad_size, i + pad_size + 1
                j_low, j_high = j - pad_size, j + pad_size + 1
                # 使用相关求取均值
                img_filter[i, j] = np.sum(img[i_low:i_high, j_low:j_high] * filter)
        img_filter = img_filter[h_low:h_high, w_low:w_high]

        return img_filter

    def rgb_mean_filter(self, pad_size=1):
        """
        对RGB通道的彩色图像进行中值滤波
        :param pad_size: - 执行滤波前在图像周围补0的维数，同时指示滤波器大小(2*pad_size + 1)
        :return: 返回滤波后的彩色图像以及各个通道分量，大小与原图像一致
        """
        b, g, r = cv2.split(self.img)
        filter = np.array([[1, 1, 1],
                           [1, 1, 1],
                           [1, 1, 1]]) / 9  # 可根据需求自己调整
        b_filter = self.gray_mean_filter(b, filter, pad_size=pad_size)
        g_filter = self.gray_mean_filter(g, filter, pad_size=pad_size)
        r_filter = self.gray_mean_filter(r, filter, pad_size=pad_size)
        img = cv2.merge((b_filter, g_filter, r_filter))
        return img, b_filter, g_filter, r_filter

    def gray_roberts_sharp(self, gray_img, throd, norm=1):
        """
        对于单通道的图像(灰度图)使用Roberts算子进行锐化
        :param gray_img: - 灰度图
        :param throd: - 锐化门限，若执行完Roberts计算后小于throd则置0
        :param norm: - 计算使用的范数类型，默认L1范数
        :return: 返回锐化后的图像，大小小于原图像
        """
        h, w = np.shape(gray_img)
        img_sharp = np.zeros((h - 1, w - 1))

        for i in range(h - 1):
            for j in range(w - 1):
                array = [gray_img[i, j] - gray_img[i + 1, j + 1],
                         gray_img[i, j + 1] - gray_img[i + 1, j]]
                img_sharp[i, j] = np.linalg.norm(array, ord=norm)

        img_sharp = np.where(img_sharp > throd, img_sharp, 0)
        return img_sharp

    def rgb_roberts_sharp(self, norm=1):
        """
        使用Roberts算子对彩色图像进行锐化
        :param norm: - 计算使用的范数类型，默认L1范数
        :return: 返回锐化后的彩色图像以及各个通道分量，大小小于原图像
        """
        b, g, r = cv2.split(self.img.copy().astype(float))
        b_sharp = self.gray_roberts_sharp(b, throd=0, norm=norm)
        g_sharp = self.gray_roberts_sharp(g, throd=0, norm=norm)
        r_sharp = self.gray_roberts_sharp(r, throd=0, norm=norm)
        img = cv2.merge((b_sharp, g_sharp, r_sharp))
        return img, b_sharp, g_sharp, r_sharp

    def gray_sobel_sharp(self, gray_img, throd, norm=1):
        """
        对于单通道的图像(灰度图)使用Sobel算子进行锐化
        :param gray_img: - 灰度图
        :param throd: - 锐化门限，若执行完Sobel计算后小于throd则置0
        :param norm: - 计算使用的范数类型，默认L1范数
        :return: 返回锐化后的图像，大小小于原图像
        """
        h, w = np.shape(gray_img)
        img_sharp = np.zeros((h - 2, w - 2))

        # 水平和垂直的Sobel算子
        r_filter = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        c_filter = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])

        for i in range(1, h - 1):
            for j in range(1, w - 1):
                array = [np.sum(gray_img[i - 1:i + 2, j - 1:j + 2] * r_filter),
                         np.sum(gray_img[i - 1:i + 2, j - 1:j + 2] * c_filter)]
                img_sharp[i - 1, j - 1] = np.linalg.norm(array, ord=norm)
        img_sharp = np.where(img_sharp > throd, img_sharp, 0)
        return img_sharp

    def rgb_sobel_sharp(self, norm=1):
        """
        使用Sobel算子对彩色图像进行锐化
        :param norm: - 计算使用的范数类型，默认L1范数
        :return: 返回锐化后的彩色图像以及各个通道分量，大小小于原图像
        """
        b, g, r = cv2.split(self.img.copy().astype(float))
        b_sharp = self.gray_sobel_sharp(b, throd=120, norm=norm)
        g_sharp = self.gray_sobel_sharp(g, throd=120, norm=norm)
        r_sharp = self.gray_sobel_sharp(r, throd=120, norm=norm)
        img = cv2.merge((b_sharp, g_sharp, r_sharp))
        return img, b_sharp, g_sharp, r_sharp

    def quick_gray_median_filter(self, gray_img, pad_size=1):
        """
        使用中值滤波的快速算法对于单通道的图像(灰度图)进行滤波
        :param gray_img: - 灰度图
        :param pad_size: - 执行滤波前在图像周围补0的维数，同时指示滤波器大小(2*pad_size + 1)
        :return: 返回滤波后的图像，大小与原图一致
        """
        img = np.pad(gray_img.copy(), pad_size, pad_with, padder=0)
        img_filter = img.copy()
        th = int((2 * pad_size + 1) ** 2 / 2) + 1  # 中间位置
        h, w = np.shape(img)
        h_low, h_high = pad_size, h - pad_size
        w_low, w_high = pad_size, w - pad_size
        global hist, mdn, ltmdn
        for i in range(h_low, h_high):
            # 统计直方图
            hist = np.zeros(256)
            pixel_list = []
            for x in range(i - pad_size, i + pad_size + 1):
                for y in range(w_low - pad_size, w_low + pad_size + 1):
                    hist[img[x, y]] += 1
                    pixel_list.append(img[x, y])
            mdn = np.sort(pixel_list)[th]
            ltmdn = 0
            # 计算中值左侧点的个数
            for pixel in pixel_list:
                if pixel < mdn:
                    ltmdn += 1
            img_filter[i, w_low] = mdn
            for j in range(w_low + 1, w_high):
                for x in range(i - pad_size, i + pad_size + 1):
                    # 减去前一个邻域最左侧的点
                    hist[img[x, j - pad_size - 1]] -= 1
                    if img[x, j - pad_size - 1] < mdn: ltmdn -= 1
                    # 加入当前邻域最右侧的点
                    hist[img[x, j + pad_size]] += 1
                    if img[x, j + pad_size] < mdn: ltmdn += 1
                # 调整中值大小
                while ltmdn < th:
                    ltmdn = ltmdn + hist[mdn]
                    mdn = mdn + 1
                while ltmdn > th:
                    mdn = mdn - 1
                    ltmdn = ltmdn - hist[mdn]
                img_filter[i, j] = mdn
        img_filter = img_filter[h_low:h_high, w_low:w_high]
        return img_filter

    def quick_rgb_median_filter(self, pad_size=1):
        """
        使用中值滤波的快速算法对RGB通道的彩色图像进行滤波
        :param pad_size: - 执行滤波前在图像周围补0的维数，同时指示滤波器大小(2*pad_size + 1)
        :return: 返回滤波后的彩色图像以及各个通道分量，大小与原图像一致
        """
        b, g, r = cv2.split(self.img)
        b_filter = self.quick_gray_median_filter(b, pad_size)
        g_filter = self.quick_gray_median_filter(g, pad_size)
        r_filter = self.quick_gray_median_filter(r, pad_size)
        img = cv2.merge((b_filter, g_filter, r_filter))
        return img, b_filter, g_filter, r_filter


# 用于np.pad
def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 10)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
    return vector


if __name__ == '__main__':
    img_path = 'lena.bmp'
    file_name = img_path.split('/')[-1]
    prefix = file_name.split('.')[0]

    img = Img(img_path)

    img_adjust_brightness = img.adjust_brightness(1.15)
    cv2.imwrite(prefix + '_adjust_brightness.bmp', img_adjust_brightness)

    img_adjust_saturation = img.adjust_saturation(1.2)
    cv2.imwrite(prefix + '_adjust_saturation.bmp', img_adjust_saturation)

    img_adjust_contrast = img.adjust_contrast(0.6)
    cv2.imwrite(prefix + '_adjust_contrast.bmp', img_adjust_contrast)

    img_adjust_hue = img.adjust_hue(0.3)
    cv2.imwrite(prefix + '_adjust_hue.bmp', img_adjust_hue)

    hist = img.get_histogram()
    plt.bar(range(len(hist)), hist)
    plt.title("histogram")
    plt.savefig(prefix + '_histogram.png')

    start = time.time()
    img_median_filter, _, _, _ = img.rgb_median_filter(pad_size=1)
    cv2.imwrite(prefix + '_median_filter.bmp', img_median_filter)
    print('普通中值滤波耗时 ', time.time() - start, 's')

    start = time.time()
    img_median_filter, _, _, _ = img.quick_rgb_median_filter(pad_size=1)
    cv2.imwrite(prefix + '_median_filter.bmp', img_median_filter)
    print('快速中值滤波耗时 ', time.time() - start, 's')

    img_mean_filter, _, _, _ = img.rgb_mean_filter(pad_size=1)
    cv2.imwrite(prefix + '_mean_filter.bmp', img_mean_filter)

    img_roberts_sharp, b, g, r = img.rgb_roberts_sharp(norm=1)
    cv2.imwrite(prefix + '_roberts_sharp.bmp', img_roberts_sharp)
    cv2.imwrite(prefix + '_roberts_sharp_b.bmp', b)
    cv2.imwrite(prefix + '_roberts_sharp_g.bmp', g)
    cv2.imwrite(prefix + '_roberts_sharp_r.bmp', r)

    img_sobel_sharp, b, g, r = img.rgb_sobel_sharp(norm=1)
    cv2.imwrite(prefix + '_sobel_sharp.bmp', img_sobel_sharp)
    cv2.imwrite(prefix + '_sobel_sharp_b.bmp', b)
    cv2.imwrite(prefix + '_sobel_sharp_g.bmp', g)
    cv2.imwrite(prefix + '_sobel_sharp_r.bmp', r)
