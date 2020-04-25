from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


class Filters:
    def __init__(self, picture_dir, picture_name):
        self.__picture_dir = picture_dir
        self.__picture_name = picture_name

    def __filter(self, kernel_size, kernel, saving_dir, normalize = False):
        image = self.__get_picture()
        img_arr = np.array(image)  # turn image into np.array
        x, y = img_arr.shape  # get the shape of the image

        pad_image = self.__padding(img_arr, x, y, kernel_size)  # padding the image
        new_image = np.zeros([x, y])  # containers to store processed image

        for lower_x in range(0, x):
            for lower_y in range(0, y):
                subimage = self.__get_subimg(lower_x, lower_y, pad_image, kernel_size)
                new_pixel = kernel(subimage)  # process the sub-image with kernel
                new_image[lower_x, lower_y] = new_pixel

        if normalize:
            new_image = self.__normalize(new_image)

        new_image = Image.fromarray(np.uint8(new_image))  # convert array to image
        new_image.save(saving_dir, 'jpeg')  # save the image

        self.print_pic(image, new_image)  # print original and processed images

    def __get_subimg(self, i, j, image, kernel_size):
        """
        :param i, j: position of starting pixel
        :param image: image to process
        :param kernel_size: size of kernel
        :return: subimage
        """
        subimg = image[i:i+kernel_size, j:j+kernel_size]
        return subimg

    def __padding(self, image, x, y, kernel_size):
        """
        :param image: array of pixels of gray-scale image
        :param x: shape of img
        :param y: shape of img
        :param kernel_size: size of kernel
        :return: array of image with padding
        """
        half = kernel_size // 2
        pad_x = x + 2 * half
        pad_y = y + 2 * half
        pad_image = np.zeros([pad_x, pad_y])
        pad_image[half:half+x, half: half+y] = image

        return pad_image

    def __get_picture(self):
        """
        open the image
        input: directory and filename of an image
        output: image object
        """
        image = Image.open(self.__picture_dir)
        image = image.convert('L')  # convert to gray_scale picture

        return image

    def __normalize(self, image):
        min_pixel = np.min(image)
        image = image - min_pixel  # set min pixel to 0

        max_pixel = np.max(image)
        image = image * (255 / max_pixel)  # set max pixel to 255

        return image

    def print_pic(self, pic1, pic2):
        """
        :param pic1: original picture
        :param pic2: transformed picture
        """
        plt.subplot(1, 2, 1)
        plt.title('Original Image')
        plt.axis('off')
        plt.imshow(pic1, cmap='gray')

        plt.subplot(1, 2, 2)
        plt.title('Processed Image')
        plt.axis('off')
        plt.imshow(pic2, cmap='gray')

    def smoothing(self):
        saving_dir = './output/smoothed_'+self.__picture_name+'.jpg'
        self.__filter(7, self.__smooth_kernel, saving_dir)

    def __smooth_kernel(self, subimage):
        """
        return the median of the pixel values
        :param subimage: a sub-image of kernel size
        :return: value of pixel
        """
        numbers = subimage.flatten()
        median = np.median(numbers)
        return median

    def sharpening(self):
        saving_dir = './output/sharpened_' + self.__picture_name + '.jpg'
        self.__filter(3, self.__Laplacian_kernel, saving_dir, normalize=True)

    def __Laplacian_kernel(self, subimage):
        laplacian = np.array([0, 1, 0, 1, -4, 1, 0, 1, 0])
        flat_img = subimage.flatten()
        pixel = flat_img[4] - 0.5 * np.sum(laplacian * flat_img)
        return pixel


if __name__ == '__main__':
    # test smoothing and sharpening
    test1 = Filters('./input/board.jpg', 'board')
    test1.smoothing()
    test2 = Filters('./input/moon.jpg', 'moon')
    test2.sharpening()


