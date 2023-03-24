import numpy as np
import matplotlib.pyplot as plt
import cv2


def histImage(im):
    h = np.zeros(256)

    for i in range(0, len(im)):
        for j in range(0, len(im)):
            h[im[i, j]] += 1
    return h


def nhistImage(im):
    h = histImage(im)
    nh = np.zeros(256)

    for i in range(0, len(h)):
        nh[i] = h[i] / np.sum(h)
    return nh


def ahistImage(im):
    ah = histImage(im)

    for i in range(1, len(ah)):
        ah[i] += ah[i - 1]
    return ah


def calcHistStat(h):
    m = np.matmul(h, range(0, len(h))) / np.sum(h)
    v = np.matmul(h, np.square(m - range(0, len(h)))) / np.sum(h)

    return m, v


def mapImage(im, tm):
    nim = im.copy()

    for i in range(0, len(tm)):
        if tm[i] < 0:
            tm[i] = 0

        if tm[i] > 255:
            tm[i] = 255

        nim[im == i] = tm[i]

    return nim


def histEqualization(im):
    ah = ahistImage(im)
    goal = np.zeros(len(ah))

    for m in range(0, len(goal)):
        goal[m] = ah[len(ah) - 1] // len(goal)
    agoal = np.cumsum(goal)

    tm = np.zeros(len(ah))

    i = 0
    j = 0

    while i < len(ah) and j < len(agoal):
        if ah[i] < agoal[j]:
            agoal[j] -= ah[i]
            tm[i] = j
            i += 1

        else:
            j += 1

    return tm