from hw2_functions import *


if __name__ == '__main__':
    wormhole = cv2.imread(r'blank_wormhole.jpg')
    im = cv2.cvtColor(wormhole, cv2.COLOR_BGR2GRAY)

    T = find_transform(src_points, dst_points)

    new_image = create_wormhole(im, T, iter=5)

    test = np.load('new_image2.npy')

    test = test - new_image
    for i in range(test.shape[0]):
        for j in range(test.shape[1]):
            if test[i][j] != 0:
                print(i, j)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(im, cmap='gray')
    plt.title('original image')
    plt.subplot(1, 2, 2)
    plt.imshow(new_image, cmap='gray')
    plt.title('wormhole image')

    plt.show()
