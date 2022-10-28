def get_threshed(image_path):

    import glob
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    import skimage.io
    import skimage.color
    import skimage.filters

    image = skimage.io.imread(image_path)
    gray_image = skimage.color.rgb2gray(image)
    blurred_image = skimage.filters.gaussian(gray_image, sigma=1.0)
    histogram, bin_edges = np.histogram(blurred_image, bins=256, range=(0.0, 1.0))
    t = 0.8
    binary_mask = blurred_image < t
    selection = image.copy()
    selection[~binary_mask] = 0
    gray_image = skimage.color.rgb2gray(image)
    blurred_image = skimage.filters.gaussian(gray_image, sigma=1.0)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)
    ret, thresh2 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY_INV)
    ret, thresh3 = cv2.threshold(img, 120, 255, cv2.THRESH_TRUNC)
    ret, thresh4 = cv2.threshold(img, 120, 255, cv2.THRESH_TOZERO)
    ret, thresh5 = cv2.threshold(img, 120, 255, cv2.THRESH_TOZERO_INV)
    return (thresh1 + thresh2 + thresh3 + thresh4 + thresh5)/5
