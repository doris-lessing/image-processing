# Image processing

Implement some classical image processing algorithms

## binarization

### 1. global thresholding
<figure class="half">
    <img src="test1.jpg", title='original image', width='20%'>
    <img src="./output_img/BGP_test1.jpg", title='processed image', width='20%'>
</figure>

### 2. local thresholding
<figure class="half">
    <img src="test1.jpg", title='original image', width='20%'>
    <img src="./output_img/LAT_test1.jpg", title='processed image', width='20%'>
</figure>

By comparing the results of global thresholding and local thresholding, we found that local thresholding includes more *details*. The local size could not be too small, otherwise the picture will be full of local details and does not make sense globally.

## interpolation
<figure class="half">
    <img src="test1.jpg", title='original image'>
    <img src="./output_img/LI_test1.jpg", title='processed image'>
</figure>
