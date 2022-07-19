# Script to apply different kinds of data augmentation to the dataset

import csv
from email.errors import FirstHeaderLineIsContinuationDefect
import os
import shutil
import cv2
from cv2 import CV_32F
from cv2 import flip
import numpy as np
from sklearn.utils import shuffle
from tqdm import tqdm
from scipy.ndimage import affine_transform

from split_data import shuffle_csv_lines

def random_flip(img, options):
    '''
    Randomly flip the image horizontally and/or vertically. 50% chance for each.
    Input: 
        image 
        options -- for determining if this augmentation is needed
    Output: flipped image
    '''
    if not options['flip']:
        return img

    # randomly flip either vertical or horizontal
    return cv2.flip(img, np.random.randint(2))

def random_rotate(img, options):
    '''
    Randomly rotate the image either 90/180/270 degrees. Equal chance for all angles.
    Input: 
        image 
        options -- for determining if this augmentation is needed
    Output: rotated image
    '''
    if not options['rotate']:
        return img

    # randomly rotate in one of three ways
    rotation_dirs = [cv2.cv2.ROTATE_90_CLOCKWISE, cv2.cv2.ROTATE_180, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE]
    choice = np.random.randint(3)
    return cv2.rotate(img, rotation_dirs[choice])

def random_crop(img, options, scale=0.85):
    '''
    Randomly crop the image s.t. crop is of scale times the size of the original image. 
    Random region is chosen as the crop and resized to match original size.
    Input: 
        image 
        options -- for determining if this augmentation is needed
        scale -- what size should the crop be.
    Output: cropped image
    '''
    if not options['crop']:
        return img

    height, width = int(img.shape[0]*scale), int(img.shape[1]*scale)
    x = np.random.randint(0, img.shape[1] - int(width))
    y = np.random.randint(0, img.shape[0] - int(height))
    cropped = img[y:y+height, x:x+width]
    return cv2.resize(cropped, (img.shape[1], img.shape[0]))


def random_colour_jitter(img, options, jitter_options=None):
    '''
    Random colour jittering of image. 
    Contains three sub functions, all or none of which may be triggered depending on jitter_options dict.
    Input: 
        image 
        options -- for determining if this augmentation is needed
        jitter_options -- for determing which sub functions triger
    Output: colour jitterd image
    '''
    if not options['jitter']:
        return img

    # deafult options is all true
    # dictionary is used to determine which jittering functions will trigger for image
    if not jitter_options:
        jitter_options = {'brt':True, 'sat':True, 'con':True}

    def random_brightness(img, jitter_options):
        '''
        Randomly jitter image brightness
        '''
        # check if needed
        if not jitter_options['brt']:
            return img

        value = np.random.randint(-40,41)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        if value >= 0:
            lim = 255 - value
            v[v > lim] = 255
            v[v <= lim] += value
        else:
            lim = np.absolute(value)
            v[v < lim] = 0
            v[v >= lim] -= np.absolute(value)

        final_hsv = cv2.merge((h, s, v))
        return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    def random_saturation(img, jitter_options):
        '''
        Randomly jitter image saturation.
        '''
        # check if needed
        if not jitter_options['sat']:
            return img

        value = np.random.randint(-40,41)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        if value >= 0:
            lim = 255 - value
            s[s > lim] = 255
            s[s <= lim] += value
        else:
            lim = np.absolute(value)
            s[s < lim] = 0
            s[s >= lim] -= np.absolute(value)

        final_hsv = cv2.merge((h, s, v))
        return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    def random_contrast(img, jitter_options):
        '''
        Randomly jitter image contrast
        '''
        # check if needed
        if not jitter_options['con']:
            return img

        # multiply by random val
        max_value = 0.4
        dummy = np.int16(img)
        dummy = dummy * np.random.uniform(1-max_value, 1+max_value)
        dummy = np.clip(dummy, 0, 255)
        return np.uint8(dummy)

    # randomly change brightness, saturation, and contrast
    # jitter_options determines which functions will actually trigger
    new_img = img.copy()
    functions = [random_brightness, random_saturation, random_contrast] 
    for function in functions:
        new_img = function(new_img, jitter_options)
    
    return new_img

def random_warp(img, options, shear_scale = 3): 
    '''
    Affine transformation (warping) of image. 
    Random shear is selected in each direction scaled by shear_scale.
    Input: 
        image 
        options -- for determining if this augmentation is needed
        shear scale 
    Output: warped image
    '''
    if not options['warp']:
        return img

    # transformation matrix with random shear values
    pt1 = np.random.uniform() / shear_scale
    pt2 = np.random.uniform() / shear_scale
    shear_M = np.array([[1,pt1,0],[pt2,1,0],[0,0,1]])

    # apply transform
    return affine_transform(img, shear_M, mode='wrap')

def gaussian_blur(img, options, kernel=5):
    '''
    Gaussian blurring of image. Uses cv2.GaussianBlur
    Input: 
        image, 
        options -- for determining if this augmentation is needed
        kernel size
    Output: blurred image
    '''
    if not options['blur']:
        return img

    # apply blur
    return cv2.GaussianBlur(img, (kernel,kernel), sigmaX=1)


def generate_new_image(img, policy, options):
    '''
    Performs the actual augmentations here based on passed policy and options
    Input: Old image, and bools for whteher to rotate and/or jitter
    Output: New image after augmnetation
    '''
    new_img = img.copy()

    # custom policy: which ones trigger depends on passed options. 
    if policy == 'custom':
        new_img = random_flip(new_img, options)
        new_img = random_rotate(new_img, options)
        new_img = random_crop(new_img, options)
        new_img = random_colour_jitter(new_img, options)
        new_img = random_warp(new_img, options)
        new_img = gaussian_blur(new_img, options)

    # simaugmnet: flips, rotate, jitter, warping, blur
    if policy == 'simaugment':
        new_img = random_flip(new_img, options)
        new_img = random_rotate(new_img, options)
        new_img = random_colour_jitter(new_img, options)
        new_img = random_warp(new_img, options)
        new_img = gaussian_blur(new_img, options)

    return new_img

def get_options_dir(options):
    '''
    Utility function for uniformly formatting augmentations options dictionary as a string. 
    Called by many other functions. The returned string is used to denote a directory name.
    Input: options dictioanry
    Returns: string representation of dictionary
    '''
    return f"f={options['flip']},r={options['rotate']},c={options['crop']},j={options['jitter']},w={options['warp']},b={options['blur']}"

def augment_data(augment_params):
    '''
    Goes through directory and adds augmented data images (all except green images) 
    Input: params dictionary
    Output: None (saves to same dir)
    '''
    # get parameters TODO: randaugment
    policy = augment_params.get('policy')
    options = augment_params.get('options')
    train_dir = augment_params.get('train_dir', './datasets/split/train')
    output_dir = augment_params.get('output_dir', f'./datasets/augmented/{policy}/{get_options_dir(options)}')

    # if no need to augment
    if policy == 'none' or not np.any(options.values()) :
        print("No need to augment data")
        return

    # overwrite augmented directory and copy over folders from split_data_dir
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    shutil.copytree(train_dir, output_dir)

    # get train image names (won't be augmenting green)
    train_names_to_augment = [name for name in os.listdir(train_dir) if name.endswith('.png') and not 'green' in name]
    green_count = len(os.listdir(train_dir)) - len(train_names_to_augment) - 1

    # loop through required labels
    lines = []
    for label in tqdm(['red', 'yellow', 'leafless']):
        idx = 0 
        naming_idx = 1
        label_names = [name for name in train_names_to_augment if label in name]
        label_count = len(label_names)

        np.random.shuffle(label_names)

        # loop through names for this label until required amount of images reached
        while label_count + naming_idx-1 < green_count:

            # check if need to loop around
            if idx >= label_count-1:
                idx = 0
                np.random.shuffle(label_names)
            else:
                idx += 1

            # load image and pass to generator
            name = label_names[idx]
            img = cv2.imread(f"{train_dir}/{name}")
            new_img = generate_new_image(img, policy, options)

            # get new name and save
            new_name = name.split('_')
            new_name[-1] = f'augmented_{naming_idx}.png'
            new_name = '_'.join(new_name)
            cv2.imwrite(f'{output_dir}/{new_name}', new_img)
            naming_idx += 1

            # add line to csv
            xmin, ymin = 0, 0
            xmax, ymax = img.shape[0], img.shape[1]
            line = f"{new_name},{xmin},{ymin},{xmax},{ymax},{label}\n"
            lines.append(line)
            
    # write new lines
    train_csv_file = open(f"{output_dir}/_train_info.csv", 'a')
    np.random.shuffle(lines)
    train_csv_file.writelines(lines)
    train_csv_file.close()

    # shuffle entire csv
    shuffle_csv_lines(f"{output_dir}/_train_info.csv")

    



if __name__ == "__main__":
    #augment_data({})

    # tests for numbers
    # train_names = [name for name in os.listdir('./datasets/augmented/(r=True,j=False)')]
    # print(len([name for name in train_names if 'green' in name]))
    # print(len([name for name in train_names if 'red' in name]))
    # print(len([name for name in train_names if 'yellow' in name]))
    # print(len([name for name in train_names if 'leafless' in name]))

    # test warping
    img = cv2.imread('datasets\split\\train\\aug100_yellow_6.png')
    new_img = random_colour_jitter(img, {'jitter':True})
    print(new_img.shape)
    cv2.imwrite('./warp_test.png', new_img)
    cv2.imwrite('./warp_test_org.png', img)










