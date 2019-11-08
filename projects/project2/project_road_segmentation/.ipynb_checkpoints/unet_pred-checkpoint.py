''' Functions for making prediction images from the U-net model '''

import numpy as np
from PIL import Image
import matplotlib.image as mpimg
import cv2 as cv2

from data_extraction import img_float_to_uint8, post_process

def get_prediction_pixel(img, model, NEW_DIM_TRAIN):
    
    # test images must be rascaled to the same size as the training images
    image = img.resize((NEW_DIM_TRAIN , NEW_DIM_TRAIN))#, refcheck=False)
    data = np.asarray(image)
    # mapping one image into 4 dimensions as training and test must have same size
    data4d = np.zeros((1,NEW_DIM_TRAIN,NEW_DIM_TRAIN,3))
    data4d[0,:,:,:] = data

    pred = model.predict(data4d)
    # scaling the image from 0 to 1 to 0 to 255
    prediction = np.multiply(pred,255.0)
    # outputting only one channel
    output_prediction = prediction[:,:,:,0]

    return output_prediction


def label_to_img_unet(imgwidth, imgheight, w, h, output_prediction,datatype):

    predict_img = np.zeros([imgwidth, imgheight,3],dtype=np.uint8)

    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            #already made black and white
            meanval = np.mean(output_prediction[j:j+w, i:i+h,0])
            if meanval>=128:
                val = 255
            else:
                val = 0
            predict_img[j:j+w, i:i+h,0] = val
            predict_img[j:j+w, i:i+h,1] = val
            predict_img[j:j+w, i:i+h,2] = val
    return predict_img
    

def make_img_overlay_pixel(img, predicted_img, PIXEL_DEPTH):
    w, h = img.size
    predicted_img = np.asarray(predicted_img)
    color_mask = np.zeros((w, h, 3), dtype=np.uint8) 
    # creating a mask of the predicted image
    color_mask[:,:,0] = predicted_img[:,:,0]

    img8 = img_float_to_uint8(img, PIXEL_DEPTH)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")

    new_img = Image.blend(background, overlay, 0.2)

    return new_img


def get_predictionimage_pixelwise(filename, image_idx, datatype, model, PIXEL_DEPTH, NEW_DIM_TRAIN):

    i = image_idx
    # Specify the path of the 
    if (datatype == 'train'):
        imageid = "satImage_%.3d" % image_idx
        image_filename = filename + imageid + ".png"
    elif (datatype == 'test'):
        imageid = "/test_%d" % i
        image_filename = filename + imageid + imageid + ".png"
    else:
        print('Error: Enter test or train')      

    # loads the image in question
    img = mpimg.imread(image_filename)
    output_prediction = get_prediction_pixel(img, model, NEW_DIM_TRAIN) #(1,400,400)
    output_prediction = np.transpose(output_prediction, (1, 2, 0)) #(400,400,1)
    predict_img = np.asarray(output_prediction)

    # Changes into a 3D array, to easier turn into image
    predict_img_3c = np.zeros((predict_img.shape[0], predict_img.shape[1], 3), dtype=np.uint8)
    predict_img8 = img_float_to_uint8(predict_img, PIXEL_DEPTH)          
    predict_img_3c[:,:,0] = predict_img8
    predict_img_3c[:,:,1] = predict_img8
    predict_img_3c[:,:,2] = predict_img8

    imgpred = Image.fromarray(predict_img_3c)
    imgpredict = imgpred.resize((608,608))

    return imgpredict


def get_pred_img_pixelwise(filename, image_idx, datatype, model, PIXEL_DEPTH, NEW_DIM_TRAIN, prediction_test_dir):

    i = image_idx
    # Specify the path of the 
    if (datatype == 'train'):
        imageid = "satImage_%.3d" % image_idx
        image_filename = filename + imageid + ".png"
    elif (datatype == 'test'):
        imageid = "/test_%d" % i
        image_filename = filename + imageid + imageid + ".png"
    else:
        print('Error: Enter test or train')      
    # loads the image in question
    img = Image.open(image_filename)
    output_prediction = get_prediction_pixel(img, model, NEW_DIM_TRAIN) #(1,224,224)
    output_prediction = np.transpose(output_prediction, (1, 2, 0)) #(224,224,1)
    predict_img = np.asarray(output_prediction)

    # Changes into a 3D array, to easier turn into image
    predict_img_3c = np.zeros((predict_img.shape[0],predict_img.shape[1], 3), dtype=np.uint8)
    predict_img8 = np.squeeze(img_float_to_uint8(predict_img, PIXEL_DEPTH))
    predict_img8[predict_img8 >= 128] = 255 
    predict_img8[predict_img8 < 128] = 0        
    predict_img_3c[:,:,0] = predict_img8
    predict_img_3c[:,:,1] = predict_img8
    predict_img_3c[:,:,2] = predict_img8

    imgpred = Image.fromarray(predict_img_3c)
    imgpredict = imgpred.resize((608,608))

    return imgpredict, img


def get_prediction_with_overlay_pixelwise(filename, image_idx, datatype, model, PIXEL_DEPTH, NEW_DIM_TRAIN,IMG_PATCH_SIZE):

    i = image_idx
    if (datatype == 'train'):
        imageid = "satImage_%.3d" % image_idx
        image_filename = filename + imageid + ".png"
    elif (datatype == 'test'):
        imageid = "/test_%d" % i
        image_filename = filename + imageid + imageid + ".png"
    else:
        print('Error: Enter test or train')

    img = Image.open(image_filename)

    # Returns a matrix with a prediction for each pixel
    output_prediction = get_prediction_pixel(img, model, NEW_DIM_TRAIN) #(1,400,400)
    output_prediction = np.transpose(output_prediction, (1, 2, 0)) #(400,400,1)

    predict_img_3c = np.zeros((output_prediction.shape[0],output_prediction.shape[1], 3), dtype=np.uint8)
    predict_img8 = np.squeeze(img_float_to_uint8(output_prediction, PIXEL_DEPTH))
    predict_img8[predict_img8 >= 128] = 255 
    predict_img8[predict_img8 < 128] = 0       
    predict_img_3c[:,:,0] = predict_img8
    predict_img_3c[:,:,1] = predict_img8
    predict_img_3c[:,:,2] = predict_img8

    newPred = label_to_img_unet(predict_img_3c.shape[0], predict_img_3c.shape[1],IMG_PATCH_SIZE, IMG_PATCH_SIZE, output_prediction,datatype)
    imgpred = Image.fromarray(newPred)
    oimg = make_img_overlay_pixel(img, imgpred, PIXEL_DEPTH)

    return oimg, imgpred


def get_postprocessed_unet(filename, image_idx, datatype):#, IMG_PATCH_SIZE):

    i = image_idx
    # Specify the path of the 
    if (datatype == 'train'):
        imageid = "satImage_%.3d" % image_idx
        image_filename = filename + imageid + ".png"
    elif (datatype == 'test'):
        imageid = "gt_pred_%d" % i
        image_filename = filename + imageid + ".png"
    else:
        print('Error: Enter test or train')      
    img = cv2.imread(image_filename, cv2.IMREAD_GRAYSCALE)
    p_img = post_process(img)

    img_post = Image.fromarray(p_img)

    return img_post