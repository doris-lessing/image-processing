#!/usr/bin/env python
# coding: utf-8

# In[16]:


import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# In[103]:


def BGT(img: str, saving_dir: str):
    '''
    input: directory and filename of an image
    output: processed image
    '''
    # convert img to array
    image = img_to_array(img)
    
    # compute the histogram of the image
    freq = frequency(image) 
    
    # compute the threshold
    threshold = BGT_threshold(freq)
    
    # process the image using the threshold
    new_image = binarization(threshold, image)
    
    # turn array into image and save
    new_image = array_to_img(new_image, saving_dir)


# In[104]:


def frequency(image: np.array):
    '''
    input: image in the form of 2-D array
    output: an array recording the frequency of pixel 0-255
    '''
    image = image.flatten() # turn the 2-D array to 1-D
    freq = np.zeros(256)
    for i in image:
        freq[i] += 1
    
    return freq


# In[105]:


def BGT_threshold(hist: np.array) -> float:
    '''
    compute threshold using histogram
    input: histogram of image, array of length 256
    output: threshold
    '''
    # find the proper inital value
    for i in range(256):
        if hist[i] > 0:
            Min = i
            break
            
    for i in range(256):
        if hist[255-i] > 0:
            Max = 255-i
            break
        
    mu = (Min + Max)/2 # inital value
    
    epsilon = 1000 # assign a large number to epsilon
    while epsilon > (Max - Min)/100:
        dark_freq = hist[0: int(mu)+1]
        dark_number = np.array(range(0, int(mu)+1))
        mu_1 = weighted_mean(dark_number, dark_freq)
        
        light_freq = hist[int(mu): 256]
        light_number = np.array(range(int(mu), 256))
        mu_2 = weighted_mean(light_number, light_freq)
        
        new_mu = (mu_1 + mu_2)/2
        epsilon = new_mu - mu
        mu = new_mu # update mu
        
    return mu


# In[106]:


def weighted_mean(numbers: np.array, freq: np.array):
    '''
    input: 1-D array of numbers and 1-D array recording the frequency
    output: the weighted mean of the group of numbers
    '''
    n = len(numbers)
    freq_sum = 0 # record the number of numbers
    Sum = 0 # sum of every number

    for i in range(n):
        Sum += freq[i] * numbers[i]
        freq_sum += freq[i]
        
    mean = Sum / freq_sum
    return mean


# In[107]:


def img_to_array(img: str):
    '''
    open the image, print it and covert it to array
    input: directory and filename of an image
    output: 2-D array of the pixels of the image
    '''
    image = Image.open(img)
    image = image.convert('L')
    plt.subplot(1,2,1)
    plt.title('original image')
    plt.axis('off')
    plt.imshow(image, cmap = 'gray')
    image = np.array(image) # turn image into np.array

    return image


# In[378]:


def array_to_img(img: np.array, saving_dir: str):
    '''
    convert an array to gray-scale image and print&save it
    input: 2-D array of pixels in the image
    output: 
    '''
    image = Image.fromarray(np.uint8(img)) # convert the array to image
    image.save(saving_dir, 'jpeg') # save the image
    # print the image
    plt.subplot(1,2,2)
    plt.title('processed image')
    plt.axis('off')
    plt.imshow(image, cmap = 'gray')


# In[109]:


def binarization(threshold: float, img: np.array, reverse = False):
    '''
    input: 
    number in [0,255]
    image: pixels in the form of 2-D array
    reverse: if True, the lighter pixel will be dark in output
    
    output: 
    processed image in the form of 2-D array
    '''
    x = img.shape[0]
    y = img.shape[1]
    if reverse == False:
        for i in range(x):
            for j in range(y):
                if img[i,j] < threshold:
                    img[i,j] = 0
                else:
                    img[i,j] = 255
    else:
        for i in range(x):
            for j in range(y):
                if img[i,j] >= threshold:
                    img[i,j] = 0
                else:
                    img[i,j] = 255
    
    return img


# In[110]:


# test the BGP algorithm on the image
BGT('test1.jpg', 'BGP_test1.jpg')


# In[381]:


# implement the locally adaptive thresholding 
# based on moving average

def LAT(img: str, saving_dir: str, n:int):
    '''
    input: directory and filename of an image, local size
    output: processed image
    '''
    # show img & convert img to array
    image = img_to_array(img)
    
    # process the image using the MA algorithm
    new_image = MA(image,n)
    
    # turn array into image and save
    new_image = array_to_img(new_image, saving_dir)
    


# In[382]:


def MA(image: np.array,n):
    '''
    input: 2-D np.array, a gray-scale image, local size
    output: 2-D np.array, the image processed by moving average algorithm
    '''

    x, y = image.shape # get the starting point of the local area
    
    new_image = np.zeros([x,y]) # container for the new image
    
    last_pixel = (0,0) # record last visited pixel
    
    initial_slice = image[0:n,0:n].flatten()
    last_s = compute_local_threshold(initial_slice, n)
    
    for i in range(x):
        if i%2 == 0: # when i is even number, moving right
                     # note that i start from 0
            for j in range(y):
                current_pixel = (i,j)
                s = update_threshold(current_pixel, last_pixel, image, x, y, n, last_s)
                new_image[i,j] = update_pixel(image[i,j], s)
                last_pixel = current_pixel
                last_s = s
                #print(s)
                
        else: # when i is even number, moving left
            for j in range(y-1,-1,-1):
                current_pixel = (i,j)
                s = update_threshold(current_pixel, last_pixel, image, x, y, n, last_s)
                new_image[i,j] = update_pixel(image[i,j], s)
                last_pixel = current_pixel
                last_s = s
                #print(s)
        #if i >=2:
            #break
    
    return new_image


# In[383]:


def compute_local_threshold(pixels, n):
    '''
    given an 1-D array of pixel values, compute the threshold
    '''
    s = 0
    for pixel in pixels:
        s += pixel / n**2
    return s


# In[384]:


def update_threshold(current_pixel, last_pixel, image: np.array, x:int, y:int, n: int, threshold):
    '''
    input: coordinate of current pixel and last pixel, image
    output: updated threshold
    '''
    #print('cur_pixel', current_pixel)
    old_line, new_line = get_line_changes(current_pixel, last_pixel, x, y, image, n)

    for pixel in old_line.flatten():
        threshold -= pixel / n**2
        #print('old',pixel)
    for pixel in new_line.flatten():
        threshold += pixel / n**2
        #print('new', pixel)
    
    return threshold


# In[385]:


def get_line_changes(current_pixel, last_pixel, x, y, image, n):
    '''
    input: coordinate of current and last pixel, size of image, image
    output: the line deleted and added in the next 
    '''
    current_boundary = get_boundary(current_pixel, x, y, n)
    last_boundary = get_boundary(last_pixel, x, y, n)
    
    del_line = np.array([])
    add_line = np.array([])
    
    if current_boundary[0] != last_boundary[0]:
        upper = max(current_boundary[0], last_boundary[0])
        lower = min(current_boundary[0], last_boundary[0])
        col = current_boundary[1]
        #print('upper_col', upper_col, 'lower_col', lower_col)
        del_line = image[lower:upper, col: col+n]
        add_line = image[upper+n-1:upper+n, col:col+n]
    
    if current_boundary[1] != last_boundary[1]:
        upper = max(current_boundary[1], last_boundary[1])
        lower = min(current_boundary[1], last_boundary[1])
        row = current_boundary[0]
        if current_pixel[0]%2 == 0:
            del_line = image[row: row+n, lower:upper]
            add_line = image[row: row+n, upper+n-1:upper+n]

        else:
            add_line = image[row: row+n, lower:upper]
            del_line = image[row: row+n, upper+n-1:upper+n]

    return del_line, add_line


# In[386]:


def get_boundary(pixel, x, y, n):
    '''
    input: the current pixel coordinate, image size, local size
    output: the coorinate of the first pixel in the local image
    '''
    half = (n-1)//2
    boundary = []
    if pixel[0] - half < 0:
        boundary.append(0)
    elif pixel[0] + half >= x:
        boundary.append(x-n)
    else:
        boundary.append(pixel[0]-half)
     
    if pixel[1] - half < 0:
        boundary.append(0)
    elif pixel[1] + half >= y:
        boundary.append(y-n)
    else:
        boundary.append(pixel[1]-half)
        
    return boundary     


# In[387]:


def update_pixel(pixel, threshold):
    '''
    given a threshold, binaralize the pixel value
    '''
    if pixel < threshold:
        pixel = 0
    else:
        pixel = 255
    return pixel


# In[388]:


# test LAT algorithm on an image
LAT('test1.jpg', 'LAT_test1.jpg',55)


# In[371]:


def LI(img: str, saving_dir: str, n: int):
    '''
    input: the dir of img, saving_dir of img
    function: enlarge the img by n times and print it
    '''
    # show img & convert img to array
    image = img_to_array(img)
    
    # process the image using the LI algorithm
    new_image = LI_compute(image, n)
    
    # turn array into image and save
    new_image = array_to_img(new_image, saving_dir)
    


# In[375]:


def LI_compute(image, n):
    '''
    input: array of old image
    output: array of new image
    '''
    x,y = image.shape
    new_x = n*(x-1)+1
    new_y = n*(y-1)+1
    new_image = np.zeros([new_x, new_y])
    for i in range(new_x-1):
        for j in range(new_y-1):
            r = j%n / n
            s = i%n / n
            old_i = i//n
            old_j = j//n
            comp_1 = s * r * image[old_i+1, old_j+1]
            comp_2 = s * (1-r) * image[old_i+1, old_j]
            comp_3 = r * (1-s) * image[old_i, old_j+1]
            comp_4 = (1-s) * (1-r) * image[old_i, old_j]
            new_image[i,j] = comp_1+comp_2+comp_3+comp_4
            
    return new_image


# In[389]:


# test LI algorithm on an image
LI('test1.jpg', 'LI_test1.jpg',3)

