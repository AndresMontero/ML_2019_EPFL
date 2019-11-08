''' Functions for creating augmented images based on the training images, '''
''' to increase the size of the training data set                         '''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import shutil
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


def augmentation(data_dir, imgDir, groundTruthDir, train_labels_filename, train_data_filename, TRAINING_SIZE, MAX_AUG, val_img = []):

    all_img = range(1,TRAINING_SIZE+1)
    train_img = np.setdiff1d(all_img, val_img)

    seed = 0
    datagenImg = ImageDataGenerator(
            rotation_range=90,
            zoom_range=0.4,
            fill_mode= 'reflect',
            width_shift_range=0.2,
            height_shift_range=0.2,
            vertical_flip=True,
            horizontal_flip=True
            )
    datagenGT = ImageDataGenerator(
            rotation_range=90,
            zoom_range=0.4,
            fill_mode= 'reflect',
            width_shift_range=0.2,
            height_shift_range=0.2,
            vertical_flip=True,
            horizontal_flip=True
            )


    if os.path.exists(imgDir):
        shutil.rmtree(imgDir)
        print("Directory " , imgDir ,  " already exists, overwritten")
    os.makedirs(imgDir)
    if os.path.exists(groundTruthDir):
        shutil.rmtree(groundTruthDir)
        print("Directory " , groundTruthDir ,  " already exists, overwritten")
    os.makedirs(groundTruthDir)

    # moving original pictures to augmentet position
    for i in train_img:
      imageid = "satImage_%.3d" % i
      image_filename = train_data_filename + imageid + ".png"
      gt_filename = train_labels_filename + imageid + ".png"
      image_dest = imgDir + "/" + imageid + ".png"
      gt_dest = groundTruthDir + "/" + imageid + ".png"
      shutil.copyfile(image_filename, image_dest)
      shutil.copyfile(gt_filename, gt_dest)
      
    # augmenting images
    for i in train_img:
      imageid = "satImage_%.3d" % i
      image_filename = train_data_filename + imageid + ".png"
      groundtruth_filename = train_labels_filename + imageid + ".png"
      trainImg = load_img(image_filename)
      trainLabel = load_img(groundtruth_filename,color_mode='grayscale')
      img_arr = img_to_array(trainImg)
      img_arr = img_arr.reshape((1,) + img_arr.shape)
      gT_arr = img_to_array(trainLabel)
      gT_arr = gT_arr.reshape((1,) + gT_arr.shape)
      j = 0
      seed_array = np.random.randint(10000, size=(1,MAX_AUG), dtype='l')
      for batch in datagenImg.flow(
        img_arr,
        batch_size=1, 
        save_to_dir=imgDir, 
        save_prefix=imageid,
        save_format='png', 
        seed=seed_array[j]):
        j +=1
        if j>=MAX_AUG:
          break
      j = 0
      for batch in datagenGT.flow(
        gT_arr,
        batch_size=1, 
        save_to_dir=groundTruthDir, 
        save_prefix=imageid,
        save_format='png', 
        seed=seed_array[j]):
        j +=1
        if j>=MAX_AUG:
          break


def sp_noise(images, s_vs_p = 0.5, amount = 0.004):

        n_img,row,col,ch = images.shape

        outs = images

        for j in range(n_img):
            out = outs[j,:,:,:]

            # Salt mode
            num_salt = np.ceil(amount * out.size * s_vs_p)
            w_ind = np.random.randint(0, col-1, int(num_salt))
            h_ind = np.random.randint(0, row-1, int(num_salt))

            out[h_ind, w_ind, :] = [0,0,0]

            # Pepper mode
            num_pepper = np.ceil(amount* out.size * (1. - s_vs_p))
            w_ind = np.random.randint(0, col-1, int(num_pepper))
            h_ind = np.random.randint(0, row-1, int(num_pepper))

            out[h_ind, w_ind, :] = [1,1,1]
            outs[j,:,:,:] = out
            
        return outs