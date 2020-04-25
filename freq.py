from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def freq_trans_strip(dir_name, Method = 1):
    if Method == 1:
        freq_trans(dir_name, erase_strips_1)
    if Method == 2:
        freq_trans(dir_name, erase_strips_2)

def freq_trans_smooth(dir_name):
    freq_trans(dir_name, smooth)


def freq_trans(dir_name, freq_filter):
    # get original image and print
    img = get_picture(dir_name)

    # zero padding
    pad_img = zero_padding(img)

    # fft transform
    fft, spec_fft = img_to_fft(pad_img)

    # filter in frequency domain and print
    fft2, spec_fft2 = freq_filter(fft)
    # print_img(fft_shift2, 'Transformed Frequency', 3)

    # transform to spacial domain and print
    new_pad_img = fft_to_img(fft2)

    # remove padding
    new_img = rm_padding(new_pad_img)

    # print images
    print_imgs([img, pad_img, spec_fft, spec_fft2, new_pad_img, new_img])


def get_picture(dir_name):
    """
    open the image
    input: directory and filename of an image
    output: image object
    """
    image = Image.open(dir_name)
    image = image.convert('L')  # convert to gray_scale picture
    return image


def zero_padding(img):
    img = np.array(img)
    x, y = img.shape
    new_img = np.zeros([2*x, 2*y])
    new_img[0:x, 0:y] = img
    return new_img


def img_to_fft(img):
    # transform img to fft
    fft = np.fft.fft2(img)
    fft_shift = np.fft.fftshift(fft)
    # compute and print magnitude spectrum
    magnitude_spectrum = spectrum(fft_shift)

    return fft_shift, magnitude_spectrum


def fft_to_img(fft):
    fft = np.fft.ifftshift(fft)
    # transform fft back to img
    new_img = np.fft.ifft2(fft)
    new_img = np.abs(new_img)

    return new_img


def spectrum(fft_shift):
    fft_shift_abs = np.abs(fft_shift)
    magnitude_spectrum = np.log(1 + fft_shift_abs)
    return magnitude_spectrum


def smooth(fft_shift):
    m, n = fft_shift.shape
    for i in range(m):
        for j in range(n):
            dis = ((i - m / 2) ** 2 + (j - n / 2) ** 2) ** 0.5
            b = 1/(1+(dis/20)**4)  ## Butterworth function
            fft_shift[i, j] *= b

    # compute magnitude spectrum
    magnitude_spectrum = spectrum(fft_shift)

    return fft_shift, magnitude_spectrum


def erase_strips_2(fft_shift):
    m, n = fft_shift.shape
    for i in range(m):
        for j in range(n):
            b1 = get_butterworth(120, 120, m, n, i, j)
            b2 = get_butterworth(280, 280, m, n, i, j)
            b3 = get_butterworth(120, 280, m, n, i, j)
            b4 = get_butterworth(280, 120, m, n, i, j)
            b5 = get_butterworth(520, 520, m, n, i, j)
            b6 = get_butterworth(520, 120, m, n, i, j)
            b7 = get_butterworth(520, 280, m, n, i, j)
            b8 = get_butterworth(280, 520, m, n, i, j)
            b9 = get_butterworth(120, 520, m, n, i, j)

            fft_shift[i, j] *= b1 * b2 * b3 * b4 * b5 * b6 * b7 * b8 * b9


    # compute magnitude spectrum
    magnitude_spectrum = spectrum(fft_shift)

    return fft_shift, magnitude_spectrum


def get_butterworth(x1, y1, m, n, i, j):
    dis1 = ((i - (m / 2 - x1)) ** 2 + (j - (n / 2 - y1)) ** 2) ** 0.5
    b1 = 1 - 1 / (1 + (dis1 / 60) ** 4)  # Butterworth function
    dis2 = ((i - (m / 2 + x1)) ** 2 + (j - (n / 2 + y1)) ** 2) ** 0.5
    b2 = 1 - 1 / (1 + (dis2 / 60) ** 4)  # Butterworth function
    dis3 = ((i - (m / 2 - x1)) ** 2 + (j - (n / 2 + y1)) ** 2) ** 0.5
    b3 = 1 - 1 / (1 + (dis3 / 60) ** 4)  # Butterworth function
    dis4 = ((i - (m / 2 + x1)) ** 2 + (j - (n / 2 - y1)) ** 2) ** 0.5
    b4 = 1 - 1 / (1 + (dis4 / 60) ** 4)  # Butterworth function
    return b1 * b2 * b3 * b4


def erase_strips_1(fft_shift):
    m, n = fft_shift.shape
    for i in range(m):
        for j in range(n):
            dis = ((i - m / 2) ** 2 + (j - n / 2) ** 2) ** 0.5
            b = 1 / (1 + (dis / 300) ** 4)  ## Butterworth function
            # power filter
            mask = 1-b
            if mask * np.abs(fft_shift[i, j]) >= 4000:
                    fft_shift[i, j] = 0

    # compute magnitude spectrum
    magnitude_spectrum = spectrum(fft_shift)

    return fft_shift, magnitude_spectrum


def rm_padding(img):
    x, y = img.shape
    new_img = img[0:int(x/2), 0:int(y/2)]
    return new_img


def print_imgs(imgs):
    titles = ['Original image', 'Padding image', 'Frequency domain',
              'Trans freq domain', 'Trans padding image', 'Trans image']
    for k in range(6):
        plt.subplot(2, 3, k+1)
        plt.imshow(imgs[k], 'gray')
        plt.title(titles[k])
        plt.axis('off')
    plt.show()


if __name__ == '__main__':
    #freq_trans_smooth('./input/test2.png')
    #freq_trans_strip('./input/freq_testimage.png')
    freq_trans_strip('./input/freq_testimage.png', 2)
