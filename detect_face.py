
# coding: utf-8

# In[4]:

import cv2
import os
import numpy as np
from tqdm import tqdm




def main(path):
    os.makedirs('crop')
    j = 1
    for i in tqdm(range(1, 220600)):

        a = path + str(i) + ".jpg"
        img = cv2.imread(a)
        #convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #detect face
        faces = faceCascade.detectMultiScale(gray, 1.3, 15)
        #crop image to face
        for (x,y,h,w) in faces:
            roi_gray = gray[y:y+h+15, x:x+w+15]
            roi_color = img[y:y+h+15, x:x+w+15]

            cv2.imwrite('crop/' + str(j) + '.jpg', roi_color)
            j = j+1

if __name__ == '__main__':
    
    #path to input images
    path = '\testtesttest'

    #haar cascade xml file
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    main(path)
    
    
        

