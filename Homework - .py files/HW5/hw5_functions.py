import cv2
import numpy as np
from scipy.signal import convolve2d as conv
from scipy.ndimage.filters import gaussian_filter as gaussian
import matplotlib.pyplot as plt

def sobel(im):
    sobel_x = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]) / 8
    sobel_y = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8

    sobel_x = abs(conv(im, sobel_x))
    sobel_y = abs(conv(im, sobel_y))

    new_im = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)

    threshold = 8.5
    new_im[new_im <= threshold] = 0
    new_im[new_im > threshold] = 255

    return new_im


def canny(im):
    blur = gaussian(im, sigma=2)
    return cv2.Canny(blur, 60, 180)


def hough_circles(im):
    im_c = im.copy()
    circles = cv2.HoughCircles(im, cv2.HOUGH_GRADIENT_ALT, 1, 20, param1=100, param2=0.9, minRadius=0, maxRadius=100)
    for i in circles[0,:]:
        cv2.circle(im_c, (int(i[0]), int(i[1])), int(i[2]), (0, 255, 255), 3)
    return im_c


def hough_lines(im):
    im_l = im.copy()

    dst = cv2.Canny(im, 200, 300)
    lines = cv2.HoughLines(dst, 1, np.pi / 180, 180)

    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))

        cv2. line(im_l, pt1, pt2, (0, 0, 255), 5)
    return im_l

