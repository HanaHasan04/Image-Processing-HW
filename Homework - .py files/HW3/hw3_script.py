from hw3_functions import *



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    im = cv2.imread(r'Images\lena.tif')
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)


    # Salt and pepper noise ----------------------------------------------------------------
    n = 50
    p = 0.3
    H, W = im.shape
    SP_images = np.zeros((n, H, W))
    # create n images with SP noise
    for i in range(n):
        SP_images[i, :, :] = add_SP_noise(im, p)

    clean_single_image = clean_SP_noise_single(SP_images[1, :, :], radius=1)
    # clean SP noise using images
    clean_multiple_images = clean_SP_noise_multiple(SP_images)

    # plot
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(SP_images[1, :, :], cmap='gray', vmin=0, vmax=255)
    plt.title('image + SP noise')
    plt.subplot(1, 3, 2)
    plt.imshow(clean_single_image, cmap='gray', vmin=0, vmax=255)
    plt.title('cleaned single image')
    plt.subplot(1, 3, 3)
    plt.imshow(clean_multiple_images, cmap='gray', vmin=0, vmax=255)
    plt.title('Cleaned using {} images'.format(n))


    # Gaussian noise ----------------------------------------------------------------
    STD=15
    gaussian_noised_im = add_Gaussian_Noise(im, s=STD)
    clean_im = clean_Gaussian_noise(gaussian_noised_im, 2, 5)
    clean_bi_im = clean_Gaussian_noise_bilateral(gaussian_noised_im, 1, 5, 25)

    # plot
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(gaussian_noised_im, cmap='gray', vmin=0, vmax=255)
    plt.title('image + Gaussian noise')
    plt.subplot(1, 3, 2)
    plt.imshow(clean_im, cmap='gray', vmin=0, vmax=255)
    plt.title('cleaned gaussian image')
    plt.subplot(1, 3, 3)
    plt.imshow(clean_bi_im, cmap='gray', vmin=0, vmax=255)
    plt.title('Cleaned using bilateral')

    plt.show()