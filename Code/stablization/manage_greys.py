def do_gray(img):
    import cv2
    import numpy as np
    from PIL import Image
    img = Image.fromarray(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])
    mask = cv.inRange(hsv, lower_blue, upper_blue)
    img = cv.bitwise_and(img,img, mask= mask)
    return img
