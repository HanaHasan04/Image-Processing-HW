from hw5_functions import *

if __name__ == "__main__":
    print("----------------------------------------------------\n")

    print("-----------------------1. Sobel----------------------\n")
    im = cv2.imread(r'balls1.tif')
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_edges = sobel(im)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(im, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 2, 2)
    plt.imshow(im_edges, cmap='gray')

    print("-----------------------2. Canny----------------------\n")
    im = cv2.imread(r'coins1.tif')
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_edges = canny(im)  # parameters fitted for coins

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(im, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 2, 2)
    plt.imshow(im_edges, cmap='gray')


    print("-----------------------3. Hough Circles----------------------\n")
    im = cv2.imread(r'coins3.tif')
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_circles = hough_circles(im)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(im, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 2, 2)
    plt.imshow(im_circles, cmap='gray')

    print("-----------------------4. Hough Lines----------------------\n")
    im = cv2.imread(r'boxOfchocolates1.tif')
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(im, cmap='gray')

    im_lines = hough_lines(im)
    plt.subplot(1, 2, 2)
    plt.imshow(im_lines, cmap='gray')

    plt.show()
