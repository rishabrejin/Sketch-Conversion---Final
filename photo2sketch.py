import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
import os

def main():
    for i in tqdm(range(1, 142964)):
        os.mkdir("crop_sketches")
        a = path + str(i) + ".jpg"
        img_gray = cv2.imread(a, cv2.IMREAD_GRAYSCALE)
        #img_gray = cv2.resize(img_gray, (100,1)

        img_gray_inv = 255 - img_gray

        img_blur = cv2.GaussianBlur(img_gray_inv, ksize=(21, 21),
                                    sigmaX=0, sigmaY=0)

        def dodgeV2(image, mask):
            return cv2.divide(image, 255-mask, scale=256)

        img_blend = dodgeV2(img_gray, img_blur)
        b = "crop_sketches/" + str(i) +".jpg"
        cv2.imwrite(b, img_blend)

if __name__ == '__main__':
    path = "crop/"
    main()