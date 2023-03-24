import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
import scipy
from scipy.signal import convolve2d

# the copy in the first lines of the function is so that you don't ruin
# the original image. it will create a new one. 

def add_SP_noise(im, p):
    sp_noise_im = im.copy()

    num_noisy_pixels = int(np.prod(im.shape) * p)

    height = random.sample(range(0, np.prod(im.shape) - 1), num_noisy_pixels)
    width = random.sample(range(0, np.prod(im.shape) - 1), num_noisy_pixels)

    # normalization:
    height = np.floor_divide(height, im.shape[0])
    width = np.floor_divide(width, im.shape[1])

    for i in range(int(num_noisy_pixels / 2)):
        sp_noise_im[height[i]][width[i]] = 0

    for i in range(int(num_noisy_pixels / 2), num_noisy_pixels):
        sp_noise_im[height[i]][width[i]] = 255

    return sp_noise_im


def clean_SP_noise_single(im, radius):
    noise_im = im.copy()
    clean_im = np.zeros((im.shape[0], im.shape[1]), dtype = np.float32)

    for i in range(im.shape[0]):
        for j in range(im.shape[1]):

            start_i = max(0, i - radius)
            end_i = min(im.shape[0], i + radius + 1)

            start_j = max(0, j - radius)
            end_j = min(im.shape[1], j + radius + 1)

            neighborhood = noise_im[start_i:end_i, start_j:end_j]

            clean_im[i][j] = np.median(neighborhood)

    return clean_im


def clean_SP_noise_multiple(images):
    clean_image = np.median(images, axis=0)
    return clean_image


def add_Gaussian_Noise(im, s):
    gaussian_noise_im = im.copy()

    noise = np.random.normal(0, s, im.shape)
    gaussian_noise_im = gaussian_noise_im + noise

    return gaussian_noise_im


def clean_Gaussian_noise(im, radius, maskSTD):
    # calculation of the filter:
    X, Y = np.meshgrid(np.arange(-radius, radius + 1), np.arange(-radius, radius + 1))
    gaussian_filter = np.power(np.e, np.negative(np.divide(np.square(X) + np.square(Y), 2 * np.square(maskSTD), dtype=np.float32)))

    # normalization:
    gaussian_filter = np.divide(gaussian_filter, np.sum(gaussian_filter))

    cleaned_im = scipy.signal.convolve2d(im, gaussian_filter, mode='same')

    return cleaned_im.astype(np.uint8)


def clean_Gaussian_noise_bilateral(im, radius, stdSpatial, stdIntensity):
    bilateral_im = im.copy()

    for i in range(im.shape[0]):
        for j in range(im.shape[1]):

            start_i = max(0, i - radius)
            end_i = min(bilateral_im.shape[0], i + radius + 1)

            start_j = max(0, j - radius)
            end_j = min(bilateral_im.shape[1], j + radius + 1)

            # edge cases:
            if start_i == 0:
                end_i = 1 + radius * 2
            if start_j == 0:
                end_j = 1 + radius * 2
            if end_i == bilateral_im.shape[0]:
                start_i = bilateral_im.shape[0] - 1 -2 * radius
            if end_j == bilateral_im.shape[1]:
                start_j = bilateral_im.shape[1] - 1 - 2 * radius

            X,Y = np.meshgrid(np.arange(start_i, end_i), np.arange(start_j, end_j))

            # calculation of window:
            window = bilateral_im[start_i:end_i, start_j:end_j]

            # calculation of gi & gs:
            gi = np.exp(np.negative(np.divide(np.square(bilateral_im[X,Y] - bilateral_im[i][j]), 2 * np.square(stdIntensity), dtype=np.float32)))
            gs = np.exp(np.negative(np.divide(np.square(X - i) + np.square(Y - j), 2 * np.square(stdSpatial), dtype=np.float32)))

            g = gi * gs

            # normalization:
            g = np.divide(g, np.sum(g))

            bilateral_im[i][j] = np.sum(g * window)

    return bilateral_im.astype(np.uint8)


