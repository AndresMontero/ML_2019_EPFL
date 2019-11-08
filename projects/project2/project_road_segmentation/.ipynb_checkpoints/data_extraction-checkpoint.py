''' Code for extracting the training and testing data                                                      '''
''' load_data_unet() loads the images pixelwise, for the U-Net model                                       '''
''' load_data_context() loads the patches of the images, with the desired context frame, for the CNN model '''
''' For both functions there is the possibility to increase the number of training data by augmentation    '''
''' Both also enable adding noise                                                                          '''


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import random
from PIL import Image
import matplotlib.image as mpimg
import cv2 as cv2
from pathlib import Path

from image_augmentation import augmentation, sp_noise


def value_to_class(v):
    foreground_threshold = 0.25 
    df = np.sum(v)
    if df > foreground_threshold:
        return [0, 1]
    else:
        return [1, 0]


def img_crop(im, w, h):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            if is_2d:
                im_patch = im[j:j+w, i:i+h]
            else:
                im_patch = im[j:j+w, i:i+h, :]
            list_patches.append(im_patch)
    return list_patches


'''UNET DATA EXTRACTION'''


def load_data_unet(train_data_filename, train_labels_filename, test_data_filename, TRAINING_SIZE, TESTING_SIZE,VALIDATION_SIZE, 
    new_dim_train,saltpepper = 0.004,augment=False,MAX_AUG=1, augImgDir='', data_dir='', groundTruthDir='',newaugment=True):
    
    idx = random.sample(range(1, 100), VALIDATION_SIZE)
    if augment == False:
        print('No augmenting of training images')
        print('\nLoading training images')

        x_train, x_val = extract_data_pixelwise(train_data_filename,TRAINING_SIZE,  'train', new_dim_train,idx)
        print('Train data shape: ',x_train.shape)

        y_train, y_val = extract_labels_pixelwise(train_labels_filename,TRAINING_SIZE, new_dim_train,idx)
        print('Train labels shape: ',y_train.shape)

    elif augment == True:
        print('Augmenting training images...')
        if newaugment == True:
            augmentation(data_dir, augImgDir, groundTruthDir, train_labels_filename, train_data_filename, TRAINING_SIZE, MAX_AUG,idx)

        x_train, y_train = extract_aug_data_and_labels_pixelwise(augImgDir)
        print('Train data shape: ',x_train.shape)
        print('Train labels shape: ',y_train.shape)
        _, x_val = extract_data_pixelwise(train_data_filename,TRAINING_SIZE, 'train', new_dim_train,idx)
        _, y_val = extract_labels_pixelwise(train_labels_filename,TRAINING_SIZE, new_dim_train,idx)

    '''adding noise'''
    x_train = sp_noise(x_train, amount=saltpepper)

    x_test,_ = extract_data_pixelwise(test_data_filename,TESTING_SIZE,  'test', new_dim_train)
    print('Test data shape: ',x_test.shape)

    road = np.sum(y_train[:,:,:,1], dtype = int)
    background = np.sum(y_train[:,:,:,0], dtype = int)
    print('Number of samples in class 1 (background): ',road)
    print('Number of samples in class 2 (road): ',background, '\n')

    return x_train, y_train, x_test, x_val, y_val


def extract_data_pixelwise(filename,num_images, datatype, new_dim_train,val_img=[]):
    ''' Extract the images into a 4D tensor [image index, y, x, channels].'''
    ''' Values are rescaled from [0, 255] down to [0.0, 1.0].'''
    ''' Dividing into training and validation set '''
    
    t_imgs = []
    v_imgs = []
    all_img = range(1,num_images+1)
    train_img = np.setdiff1d(all_img, val_img)

    for i in train_img:
        if datatype == 'train':
            imageid = "satImage_%.3d" % i
            image_filename = filename + imageid + ".png"
        elif datatype == 'test':
            imageid = "/test_%d" % i
            image_filename = filename + imageid + imageid + ".png"
        
        if os.path.isfile(image_filename):
            # Add the image to the imgs-array
            img = Image.open(image_filename)
            img = np.asarray(img)
            t_imgs.append(img)
        else:
            print ('File ' + image_filename + ' does not exist')

    if datatype == 'train':
        for i in val_img:
            imageid = "satImage_%.3d" % i
            image_filename = filename + imageid + ".png"
            if os.path.isfile(image_filename):
                img = Image.open(image_filename)
                img = np.asarray(img)
                v_imgs.append(img)
            else:
                print ('File ' + image_filename + ' does not exist')
    
    t_arr =np.array(t_imgs)
    v_arr = np.array(v_imgs) 

    return t_arr,v_arr


def extract_labels_pixelwise(filename, num_images, new_dim_train, val_img=[]):
    '''Extract the labels into a 1-hot matrix [image index, label index].'''
    '''We want the images with depth = 2, one for each class, one of the depths is 1 and the other 0'''
    ''' dividing the labels into training set and validation set'''
    
    t_imgs = []
    v_imgs = []
    all_img = range(1,num_images+1)
    train_img = np.setdiff1d(all_img, val_img)

    for i in train_img:
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        
        if os.path.isfile(image_filename):
            # Add the image to the imgs-array
            img = Image.open(image_filename)
            img = np.asarray(img)
            t_imgs.append(img)
        else:
            print ('File ' + image_filename + ' does not exist')

    for i in val_img:
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        
        if os.path.isfile(image_filename):
            # Add the image to the imgs-array
            img = Image.open(image_filename)
            img = np.asarray(img)
            v_imgs.append(img)
        else:
            print ('File ' + image_filename + ' does not exist')

    t_imgs = np.array(t_imgs)
    v_imgs = np.array(v_imgs)
    num_t_images = len(t_imgs)
    num_v_images = len(v_imgs)

    t_labels = np.zeros((num_t_images,new_dim_train,new_dim_train,2))
    v_labels = np.zeros((num_v_images,new_dim_train,new_dim_train,2))

    foreground_threshold = 0.5
    t_labels[t_imgs > foreground_threshold] = [1,0]
    t_labels[t_imgs <= foreground_threshold] = [0,1]

    v_labels[v_imgs > foreground_threshold] = [1,0]
    v_labels[v_imgs <= foreground_threshold] = [0,1]

    # Convert to dense 1-hot representation.
    return t_labels.astype(np.float32),v_labels.astype(np.float32)


def extract_aug_data_and_labels_pixelwise(filename):
    '''Extract the images into a 4D tensor [image index, y, x, channels]'''
    '''Values are rescaled from [0, 255] down to [0.0, 1.0].'''

    imgs = []
    gt_imgs = []
    pathlist = Path(filename).glob('**/*.png')
    
    # goes through all the augmented images in image directory
    # must pair them with all the augmented groundtruth images
    # and therefore we finds the corresponding string of the groundtruth name
    
    for path in pathlist:
        # because path is object not string
        image_path = str(path)
        lhs,rhs = image_path.split("/images")
        # colored image
        img = Image.open(image_path)
        img = np.asarray(img)
        imgs.append(img)
        gt_path = lhs + '/groundtruth' + rhs
        # groundtruth image
        g_img = Image.open(gt_path)
        g_img = np.asarray(g_img)
        gt_imgs.append(g_img)

    num_images = len(imgs)
    gt_imgs = np.array(gt_imgs)
    IMG_WIDTH = imgs[0].shape[0]
    IMG_HEIGHT = imgs[0].shape[1]
    # creating labels from the groundtruths
    labels = np.zeros((num_images,IMG_WIDTH,IMG_HEIGHT,2))
    foreground_threshold = 0.5
    labels[gt_imgs > foreground_threshold] = [1,0]
    labels[gt_imgs <= foreground_threshold] = [0,1]
    imgarr = np.array(imgs)

    return imgarr, labels.astype(np.float32)



'''CNN DATA EXTRACTION'''

def load_data_context(train_data_filename, train_labels_filename, test_data_filename, TRAINING_SIZE, VALIDATION_SIZE, IMG_PATCH_SIZE, CONTEXT_SIZE, TESTING_SIZE, saltpepper = 0.004, augment=False, MAX_AUG=1, augImgDir='', data_dir='', groundTruthDir='', newaugment=True):

    # Takes a number equal to VALIDATION_SIZE of random images into the validation set
    idx = random.sample(range(1, 100), VALIDATION_SIZE)

    if augment == False:
        print('No augmenting of training images')
        print('Loading training images')
        x_train, x_val = extract_data_context(train_data_filename, TRAINING_SIZE, IMG_PATCH_SIZE, CONTEXT_SIZE,  'train', idx)

        print('Loading training labels')
        y_train, y_val = extract_labels_context(train_labels_filename, TRAINING_SIZE, IMG_PATCH_SIZE, idx)

    elif augment == True:
        print('Augmenting training images...')
        if newaugment == True:
            augmentation(data_dir, augImgDir, groundTruthDir, train_labels_filename, train_data_filename, TRAINING_SIZE, MAX_AUG, idx)
        x_train, y_train = extract_aug_data_and_labels_context(augImgDir, TRAINING_SIZE*(MAX_AUG+1), IMG_PATCH_SIZE, CONTEXT_SIZE)
        _, x_val = extract_data_context(train_data_filename, TRAINING_SIZE, IMG_PATCH_SIZE, CONTEXT_SIZE,  'train', idx)
        _, y_val = extract_labels_context(train_labels_filename, TRAINING_SIZE, IMG_PATCH_SIZE, idx)

    # Add noise to the training set
    x_train = sp_noise(x_train, amount=saltpepper)

    print('Loading test images\n')
    x_test, _ = extract_data_context(test_data_filename,TESTING_SIZE, IMG_PATCH_SIZE, CONTEXT_SIZE, 'test')

    print('Train data shape: ',x_train.shape)
    print('Train labels shape: ',y_train.shape)
    print('Test data shape: ', x_test.shape)

    return x_train, y_train, x_test, x_val, y_val


def value_to_class_context(patch, IMG_PATCH_SIZE, CONTEXT_SIZE):
    # Converts the values for the patch into a single class
    patch_center = patch[CONTEXT_SIZE:CONTEXT_SIZE+IMG_PATCH_SIZE,CONTEXT_SIZE:CONTEXT_SIZE+IMG_PATCH_SIZE]
    v = np.mean(patch)

    foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch
    df = np.sum(v)
    if df > foreground_threshold:
        return [0, 1]
    else:
        return [1, 0]


def extract_data_context(filename, num_images, IMG_PATCH_SIZE, CONTEXT_SIZE, datatype, val_img=[]):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    t_imgs = []
    v_imgs = []
    all_img = range(1,num_images+1)
    train_img = np.setdiff1d(all_img, val_img)

    for i in train_img:
        if datatype == 'train':
            imageid = "satImage_%.3d" % i
            image_filename = filename + imageid + ".png"
        elif datatype == 'test':
            imageid = "/test_%d" % i
            image_filename = filename + imageid + imageid + ".png"
        if os.path.isfile(image_filename):
            img = mpimg.imread(image_filename)
            t_imgs.append(img)
        else:
            print ('File ' + image_filename + ' does not exist')

    if datatype == 'train':
        for i in val_img:
            imageid = "satImage_%.3d" % i
            image_filename = filename + imageid + ".png"
            if os.path.isfile(image_filename):
                img = mpimg.imread(image_filename)
                v_imgs.append(img)
            else:
                print ('File ' + image_filename + ' does not exist')

    num_t_images = len(t_imgs)
    num_v_images = len(v_imgs)
    IMG_WIDTH = t_imgs[0].shape[0]
    IMG_HEIGHT = t_imgs[0].shape[1]

    # Makes a list of all patches for the image at each index
    train_img_patches = [img_crop_context(t_imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE, CONTEXT_SIZE) for i in range(num_t_images)]
    val_img_patches = [img_crop_context(v_imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE, CONTEXT_SIZE) for i in range(num_v_images)]
    # "Unpacks" the vectors for each image into a shared vector, where the entire vector for image 1 comes
    # before the entire vector for image 2
    train_data = [train_img_patches[i][j] for i in range(len(train_img_patches)) for j in range(len(train_img_patches[i]))]
    val_data = [val_img_patches[i][j] for i in range(len(val_img_patches)) for j in range(len(val_img_patches[i]))]
    
    return np.asarray(train_data), np.asarray(val_data)



def extract_aug_data_and_labels_context(filename, num_images, IMG_PATCH_SIZE, CONTEXT_SIZE, val_img=[]):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [0.0, 1.0].
    """
    imgs = []
    gt_imgs = []

    glob_path = Path(filename)
    pathlist = [str(pp) for pp in glob_path.glob('**/*.png')]
    
    #goes through all the augmented images in image directory
    # must pair them with all the augmented groundtruth images
    for path in pathlist:
        image_path = str(path)
        lhs,rhs = image_path.split("/images")
        img = mpimg.imread(image_path)
        imgs.append(img)
        gt_path = lhs + '/groundtruth' + rhs
        g_img = mpimg.imread(gt_path)
        gt_imgs.append(g_img)

    num_images = len(imgs)
    IMG_WIDTH = imgs[0].shape[0]
    IMG_HEIGHT = imgs[0].shape[1]
    N_PATCHES_PER_IMAGE = (IMG_WIDTH/IMG_PATCH_SIZE)*(IMG_HEIGHT/IMG_PATCH_SIZE)
    img_patches = [img_crop_context(imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE, CONTEXT_SIZE) for i in range(num_images)]
    
    # "Unpacks" the vectors for each image into a shared vector, where the entire vector for image 1 comes
    # befor the entire vector for image 2
    data = [img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))]
    gt_patches = [img_crop(gt_imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(num_images)]
    data_gt = np.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])
    labels = np.asarray([value_to_class(np.mean(data_gt[i])) for i in range(len(data_gt))])

    return np.asarray(data), labels.astype(np.float32)


def extract_labels_context(filename, num_images, IMG_PATCH_SIZE, val_img=[]):
    """Extract the labels in the training set into a 1-hot matrix [image index, label index]."""
    
    t_imgs = []
    v_imgs = []
    all_img = range(1,num_images+1)
    train_img = np.setdiff1d(all_img, val_img)

    for i in train_img:
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            img = mpimg.imread(image_filename)
            t_imgs.append(img)
        else:
            print ('File ' + image_filename + ' does not exist')

    for i in val_img:
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            img = mpimg.imread(image_filename)
            v_imgs.append(img)
        else:
            print ('File ' + image_filename + ' does not exist')


    num_t_images = len(t_imgs)
    num_v_images = len(v_imgs)
    t_patches = [img_crop(t_imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(num_t_images)]
    v_patches = [img_crop(v_imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(num_v_images)]

    t_data = np.asarray([t_patches[i][j] for i in range(len(t_patches)) for j in range(len(t_patches[i]))])
    v_data = np.asarray([v_patches[i][j] for i in range(len(v_patches)) for j in range(len(v_patches[i]))])
    

    t_labels = np.asarray([value_to_class(np.mean(t_data[i])) for i in range(len(t_data))])
    v_labels = np.asarray([value_to_class(np.mean(v_data[i])) for i in range(len(v_data))])

    # Convert to dense 1-hot representation.
    return t_labels.astype(np.float32), v_labels.astype(np.float32)


def img_crop_context(im, w, h, w_context):
    #Creates a list with patches with the context surrounding the pixel
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    
    #Creates the image with reflected edges
    im_reflect = cv2.copyMakeBorder(im, w_context, w_context, w_context, w_context, cv2.BORDER_REFLECT)

    for i in range(0+w_context,imgheight+w_context,h): # iterates through the 0th axis
        for j in range(0+w_context,imgwidth+w_context,w): # iterates through the 1th axis
            l = j - w_context # left
            r = j + w + w_context # right
            t = i - w_context # bottom
            b = i + h + w_context # top
            
            if is_2d:
                im_patch = im_reflect[l:r, t:b]
            else:
                im_patch = im_reflect[l:r, t:b, :]
            list_patches.append(im_patch)
    return list_patches


def img_float_to_uint8(img, PIXEL_DEPTH):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * PIXEL_DEPTH).round().astype(np.uint8)
    return rimg


def post_process(img):
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT,(33,1))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(1,33))
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT,(17,1))
    kernel4 = cv2.getStructuringElement(cv2.MORPH_RECT,(1,17))

    img1 = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel1)
    img2 = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel2)

    img_open = cv2.bitwise_or(img1, img2)

    img3 = cv2.morphologyEx(img_open, cv2.MORPH_CLOSE, kernel3)
    img4 = cv2.morphologyEx(img_open, cv2.MORPH_CLOSE, kernel4)
    
    img_close = cv2.bitwise_or(img3, img4)
    
    return img_close
