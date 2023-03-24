from hw1_123456789 import *

if __name__ == '__main__':
    # read the images you want to try on - in this example I'm using the four squares
    # image because it is easy to predict what the histogram will be
    image = cv2.imread(r"Images\barbara.tif")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # open new window
    plt.figure(0)
    # show the image you're working with
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    # give the plot a title
    plt.title('original image')

    # 1 - check histImage function
    # h is the histogram of the image
    h = histImage(image)
    # plot the histogram
    plt.figure(1)
    plt.plot(h)
    plt.title('histogram of image')

    # 2 - check nhistimage function
    # nh is the normalized histogram of the image
    nh = nhistImage(image)
    # plot the histogram
    plt.figure(2)
    plt.plot(nh)
    plt.title('normalized histogram of image')

    # 3 - check ahistImage function
    # ah is the accumulative histogram of the image
    ah = ahistImage(image)
    # plot the histogram
    plt.figure(3)
    plt.plot(ah)
    plt.title('accumulative histogram of image')

    # 4 - check calcHistStat function
    # calculate mean and variance of histogram of the image
    m, v = calcHistStat(h)
    print("mean = {}, variance ={}".format(m,v))

    # 5 - check mapImage function
    # map an image using a tone mapping
    # examples of tone mapping:
    tm_brighter = 2 * np.arange(256) + 10
    tm_darker = 1/2 * np.arange(256)
    tm_negative = 255 - np.arange(256)
    nm = mapImage(image, tm_darker)
    # plot
    plt.figure(5)
    # original image
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    plt.title('original image')
    # mapped image - should be darker/brighter/negative based on the tm you used
    plt.subplot(1, 2, 2)
    plt.imshow(nm, cmap='gray', vmin=0, vmax=255)
    plt.title('mapped image')

    # 6 - check histEqualization function
    # calculate the tone mapping that maps im1 to an equalized histogram image
    im = cv2.imread(r"Images\darkimage.tif")
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    tm_equalization = histEqualization(im)
    # to check if we had it correctly
    # apply the tone mapping to the image
    mappedImage = mapImage(im, tm_equalization)
    # plot
    plt.figure(6)
    # original image
    plt.subplot(2, 2, 1)
    plt.imshow(im, cmap='gray', vmin=0, vmax=255)
    plt.title('before equalization')
    # equalized image
    plt.subplot(2, 2, 2)
    plt.imshow(mappedImage, cmap='gray', vmin=0, vmax=255)
    plt.title('after equalization')
    # histogram of original image
    plt.subplot(2, 2, 3)
    plt.plot(histImage(im))
    # histogram of equalized image
    plt.subplot(2, 2, 4)
    plt.plot(histImage(mappedImage))
    # Do you see a brighter image and more equalized histogram??
    # if not, something is wrong with your equalization function

    plt.show()