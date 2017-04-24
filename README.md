# Sketch-Conversion---Final
We aim to convert sketches of face images to their photo realistic versions

This project aims to construct a photo-realistic version of hand-drawn sketches of human faces using Deep Convolutional Neural Networks. 

 Additionally, we also propose to use our trained network for the purpose of colorization of grey-scale images and videos.
 
 The sample images are provided in this repository.
 
 detect_faces.py is used to run a Haar cascade over the dataset of images to detec and crop face images and align them such that they can be fed into the network. The path where training images lie need to be specified in the file as 'path'
 
 fetch_data.py is the python file that converts the training input and output images into a numpy array and saves as two .npy files; one each for input and output. Here both the input and output images are downsized to 100 * 100 pixels. The directory for input and output images need to specified.
 
 live_video_output.py loads a saved model or trains a model from train_model.py and a real-time grayscale video captured from the webcam is converted to a colorized version using the trained model in real-time.
 
 To predict a model run python model_predict.py <image_name>. This file saves the output photo-realistic image from an input sketch using an imported model that has either been saved earlier or trains a network to do the same.
 
 photo2sketch.py is the code used to convert the photo realistic images to their respective sketches to generate the training input data. The OpenCV library is used to implement this. The path where the images are saved need to be specified in the file.
 
 train_model.py returns a saved model that is loaded, or executes the entire training process if a trained model is not found. TFlearn abstraction layers are used to train the model and will run for 5 epochs.
 
 
