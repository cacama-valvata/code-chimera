#! /usr/bin/env python3

from PIL import Image

def cleanup_shadow(img):
    src = img.copy()
    kernel_7x7 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    """https://stackoverflow.com/questions/19363293/whats-the-fastest-way-to-increase-color-image-contrast-with-opencv-in-python-c"""
    """https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html"""
    gamma, phi, theta = 1.5, 1.0, 1.0
    # build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
    lookuptable = np.arange(256)
    # apply gamma correction using the lookup table
    lookUpTable = (np.power(lookuptable / 255.0, gamma) * 255).astype(np.uint8)
    lookUpTable[235:] = 255
    
    rows, cols = src.shape[:2]
    width = int(cols*500.0/rows)
    dst = cv2.resize(src,(width, 500))
 

    # 2) Dilate the image to preserve the bar code
    dst = cv2.threshold(dst, 250, 0, cv2.THRESH_TRUNC)[1]
    # dst = cv2.bilateralFilter(dst,7,7,7)
    dilated = cv2.dilate(dst, kernel_7x7, iterations=1)
    # _imwrite(imfile,dilated)

    # 3) Blur the result to further suppress any text.
    background = cv2.medianBlur(dilated, 9)
    # background = cv2.GaussianBlur(dilated, (7, 7),1.0)
    #    _imwrite(imfile, background)

    background = cv2.resize(background, (cols, rows))

    # 4) Calculate the difference betweem the original
    #    and the image we just get. the identical will
    #    be black(close to 0), the text will be white(large).
    thresh = 255 - cv2.absdiff(src, background)
    #    _imwrite(imfile, thresh)

    # 5) Normalize the image.
    """https://zh.wikipedia.org/wiki/Lab%E8%89%B2%E5%BD%A9%E7%A9%BA%E9%97%B4"""
    lab = cv2.cvtColor(thresh, cv2.COLOR_BGR2LAB)
    lum = lab[:, :, 0]  # luminance

    # 6) Re-normalize the image so that we can almose clear the gray as much as we can.
    # reduce luminosity
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(13, 13))
    """https://stackoverflow.com/questions/19363293/whats-the-fastest-way-to-increase-color-image-contrast-with-opencv-in-python-c"""
    """https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html"""
    lum = clahe.apply(lum)  # apply CLAHE to the L-channel
    lab[:, :, 0] = lum
    thresh = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    thresh = cv2.LUT(thresh, lookUpTable)


    return background, thresh

image_file = Image.open("pic/wave3.png") # open colour image
image_file = image_file.convert('1') # convert image to black and white
image_file.save('wave3_bw.png')




