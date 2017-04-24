
# coding: utf-8

# In[1]:

import numpy as np
import cv2
import tensorflow as tf
tf.reset_default_graph()


import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d, conv_2d_transpose
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import batch_normalization


import os
from tqdm import tqdm
import train_model
def main():
    #loading pre-trained model or train model if model does not exist 
    model = train_model.model()

    cap = cv2.VideoCapture(0)
    #cap.set(6, 10)

    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(gray, 1.3, 15)

        for (x,y,h,w) in faces:
            roi_gray = gray[y:y+h+25, x:x+w+25]
            roi_color = frame[y:y+h+25, x:x+w+25]

            #grayscale video
            frame = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
            img_gray = np.dstack([frame, frame, frame])
            
            #converts to sketch
            img_gray_inv = 255 - img_gray

            img_blur = cv2.GaussianBlur(img_gray_inv, ksize=(21, 21),
                                        sigmaX=0, sigmaY=0)

            def dodgeV2(image, mask):
                return cv2.divide(image, 255-mask, scale=256)

            img_blend = dodgeV2(img_gray, img_blur)
            test_data = cv2.resize(img_blend, (100,100))
            #img_blend = cv2.resize(img_blend, (100,100))

            photo = cv2.resize(roi_color, (100,100))

            cv2.imshow('frame', test_data)

            #cv2.imshow('frame2', photo)
            #output prediction of model
            model_out = model.predict([test_data])
            
            #normalize frame output
            x = np.reshape(model_out, (100,100,3))
            maxVal = max(x.flatten());
            minVal = min(x.flatten());

            Img = (((x.flatten() - minVal) / ( maxVal - minVal)))

            x = np.reshape(Img, (100,100,3))
            
            cv2.imshow('frame3', x)
            


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

