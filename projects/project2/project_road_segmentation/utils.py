import scipy as scipy
from tensorflow.keras.utils import to_categorical
import cv2
import matplotlib.image as mpimg
import numpy as np
from PIL import Image
import re

def load_img(filename):
    """Load image from directory.
    Args:
        filename (string): path of the directory
    Returns:
       img (numpy.float64): img loaded
    """
    img = mpimg.imread(filename)
    return img

def crop_image(img, w, h):
    """Divides img and returns the patches.
    Args:
        img (image): Input image
        w (numpy.int64): the width of the image
        h (numpy.int64): the height of the image
    Returns:
       list_Patches (list[]): list of patches
    """
    list_Patches = []
    img_width = img.shape[0]
    img_height = img.shape[1]
    for i in range(0,img_height,h):
        for j in range(0,img_width,w):
            if len(img.shape) < 3:
                im_patch = img[j:j+w, i:i+h]
            else:
                im_patch = img[j:j+w, i:i+h, :]
            list_Patches.append(im_patch)
    return list_Patches

def pad_matrix(mat, h_pad, w_pad, val=0):
    """Add padd to images.
    Args:
        mat (image): part of the image to add padding
        h_pad (numpy.int64): the height of the padding
        w_pad (numpy.int64): the width of the padding
    Returns:
       padded_mat (image): part of the image with padding
    """
    h_pad = int(h_pad)
    w_pad = int(w_pad)
    if len(mat.shape) == 3:
        padded_mat = np.pad(
            mat,
            ((h_pad, h_pad), (w_pad, w_pad), (0, 0)),
            mode="constant",
            constant_values=((val, val), (val, val), (0, 0)),
        )
    elif len(mat.shape) == 2:
        padded_mat = np.pad(
            mat,
            ((h_pad, h_pad), (w_pad, w_pad)),
            mode="constant",
            constant_values=((val, val), (val, val)),
        )
    else:
        raise ValueError("This method can only handle 2d or 3d arrays")
    return padded_mat


def imag_rotation(X, Y, number_rotations=8):
    """Rotates images for data augmentation.
    Args:
        X (image): part of the image to rotate
        Y (image): part of the groundtruth to rotate
        number_rotations (numpy.int64): the times to rotate image
    Returns:
        Xrs (image): part of the image rotated
        Yrs (image): part of the groundtruth rotated
    """
    w = X.shape[1]
    w_2 = w // 2  # half of the width
    padding = 82
    padding2 = 24
    Xrs = X
    Yrs = Y

    ### Add padding2
    Xrs = pad_matrix(Xrs, padding2, padding2)
    Yrs = pad_matrix(Yrs, padding2, padding2)
    ###

    Xrs = np.expand_dims(Xrs, 0)
    Yrs = np.expand_dims(Yrs, 0)

    thetas = np.random.randint(0, high=360, size=number_rotations)
    for theta in thetas:
        Xr = pad_matrix(
            X, padding, padding
        )  # Selected for the specific case of images of (400,400)
        Yr = pad_matrix(
            Y, padding, padding
        )  # Selected for the specific case of images of (400,400)
        Xr = scipy.ndimage.rotate(Xr, theta, reshape=False)
        Yr = scipy.ndimage.rotate(Yr, theta, reshape=False)
        theta = theta * np.pi / 180
        a = int(
            w_2 / (np.sqrt(2) * np.cos(np.pi / 4 - np.mod(theta, np.pi / 2)))
        )  # width and height of the biggest square inside the rotated square
        w_p = w_2 + padding
        Xr = Xr[w_p - a : w_p + a, w_p - a : w_p + a, :]
        Yr = Yr[w_p - a : w_p + a, w_p - a : w_p + a]

        Xr = cv2.resize(Xr, dsize=(w_2 * 2, w_2 * 2), interpolation=cv2.INTER_CUBIC)
        Yr = cv2.resize(Yr, dsize=(w_2 * 2, w_2 * 2), interpolation=cv2.INTER_CUBIC)

        if np.random.choice(2) == 1:
            Xr = np.flipud(Xr)
            Yr = np.flipud(Yr)

        if np.random.choice(2) == 1:
            Xr = np.fliplr(Xr)
            Yr = np.fliplr(Yr)

        ### Add padding2
        Xr = pad_matrix(Xr, padding2, padding2)
        Yr = pad_matrix(Yr, padding2, padding2)
        ###

        Xr = np.expand_dims(Xr, 0)
        Yr = np.expand_dims(Yr, 0)
        Xrs = np.append(Xrs, Xr, axis=0)
        Yrs = np.append(Yrs, Yr, axis=0)

    return Xrs, Yrs


def imag_rotation_aug(Xr, Yr, number_rotations=8):
    """Data augmentation with rotation of images.
    Args:
        Xr (image): part of the image rotated
        Yr (image): part of the groundtruth rotated
        number_rotations (numpy.int64): the times to rotate image
    Returns:
        Xrs_shuf (image): part of the image rotated randomly
        Yrs_shuf (image): part of the groundtruth rotated randomly
    """
    Xrs, Yrs = imag_rotation(Xr[0], Yr[0])
    for i in range(1, len(Xr)):
        Xrr, Yrr = imag_rotation(Xr[i], Yr[i])
        Xrs = np.append(Xrs, Xrr, axis=0)
        Yrs = np.append(Yrs, Yrr, axis=0)

    Xrs_shuf = []
    Yrs_shuf = []
    index_shuf = list(range(len(Xrs)))
    np.random.shuffle(index_shuf)
    for i in index_shuf:
        Xrs_shuf.append(Xrs[i])
        Yrs_shuf.append(Yrs[i])
    # Add padding to the original images to validate borders

    return Xrs_shuf, Yrs_shuf

def create_minibatch(X, Y, n, w_size=64, batch_size=250, patch_size=16, width = 448):
    """Creates Minibatch to pass to the generator of the model .
    Args:
        X (image): Images features
        Y (image): Groundtruth images
        w_size (numpy.int64): window size 
        batch_size (numpy.int64): batch size to train the model 
        patch_size (numpy.int64): size of the patches
    Yields:
        batch_images (images): batch of images to train
        batch_labelss (list[int]): labels of the images in the batch
    """
    num_images = n
    w_size = w_size
    batch_size = batch_size
    patch_size = patch_size

    while True:
        batch_images = np.empty((batch_size, w_size, w_size, 3))
        batch_labels = np.empty((batch_size, 2))
        for i in range(batch_size):
            # Select a random index representing an image
            random_index = np.random.choice(num_images)
            # Width of original image
            width = width
            # Sample a random window from the image
            random_sample = np.random.randint(w_size // 2, width - w_size // 2, 2)
            # Create a sub image of size w_size x w_size
            sampled_image = X[random_index][
                random_sample[0] - w_size // 2 : random_sample[0] + w_size // 2,
                random_sample[1] - w_size // 2 : random_sample[1] + w_size // 2,
            ]
            # Take its corresponding ground-truth image
            correspond_ground_truth = Y[random_index][
                random_sample[0] - patch_size // 2 : random_sample[0] + patch_size // 2,
                random_sample[1] - patch_size // 2 : random_sample[1] + patch_size // 2,
            ]

            # We set in the label depending on the threshold of 0.25
            # The label becomes either 0 or 1 by applying to_categorical with parameter 2
            label = to_categorical(
                (np.array([np.mean(correspond_ground_truth)]) > 0.25) * 1, 2
            )

            # We put in the sub image and its corresponding label before yielding it
            batch_images[i] = sampled_image
            batch_labels[i] = label
        yield (batch_images, batch_labels)

def create_patches(X, patch_size, stride, padding):
    """Creates Patches to classify.
    Args:
        X (image): Images features
        patch_size (numpy.int64): patch size
        stride (numpy.int64): stride 
        padding (numpy.int64): padding
    Returns:
        img_patches (images): patches of the image
        batch_labelss (list[int]): labels of the images in the batch
    """
    img_patches = np.asarray([cut_image_for_submission(X[i], patch_size, patch_size, stride, padding) for i in range(X.shape[0])])
    img_patches = img_patches.reshape(-1, img_patches.shape[2], img_patches.shape[3], img_patches.shape[4])
    
    return img_patches 


def cut_image_for_submission(img, w, h, stride, padding):
    """Cuts images in patches to submit to Aicrowd.
    Args:
        img (image): Images features
        w (numpy.int64): width
        h (numpy.int64): height
        stride (numpy.int64): stride 
        padding (numpy.int64): padding
    Returns:
        list_Patches : list of the patches to submit
    """
    list_Patches = []
    width = img.shape[0]
    height = img.shape[1]
    img = np.lib.pad(img, ((padding, padding), (padding, padding), (0,0)), 'reflect')
    for i in range(padding,height+padding,stride):
        for j in range(padding,width+padding,stride):
            img_patch = img[j-padding:j+w+padding, i-padding:i+h+padding, :]
            list_Patches.append(img_patch)
    return list_Patches

# assign a label to a patch
def patch_to_label(patch):
    """Classifies patch to a label 0 or 1.
    Args:
        patch (image): patch of image
    Returns:
        w (numpy.int64): width
    """
    foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch
    df = np.mean(patch)
    if df > foreground_threshold:
        return 1
    else:
        return 0

def mask_to_submission_strings(model, filename, patch_size = 16):
    """Classifies the images of the test set for the submisiion.
    Args:
        model (model): trained Model
        filename (string): path of the image
    Yields:
        Labels for patches of the image
    """
    img_number = int(re.search(r"\d+", filename).group(0))
    img = load_image(filename)
    img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
    labels = model.classify(img)
    labels = labels.reshape(-1)
    count = 0
    print("Processing image => " + filename)
    for j in range(0, img.shape[2], patch_size):
        for i in range(0, img.shape[1], patch_size):
            label = int(labels[count])
            count += 1
            yield("{:03d}_{}_{},{}".format(img_number, j, i, label))

# def group_patches(patches, num_images):
#     return patches.reshape(num_images, -1)

# Create the csv file
def generate_submission(model, submission_filename, *image_filenames):
    """ Generate a .csv file with the classification of the imges of the test set
    Args:
        model (model): Trained Model
        submission_filename (string): path of the image
        *image_filenames : list of strings
    """
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for fn in image_filenames[0:]:
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(model, fn))


