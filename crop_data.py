# Script for cropping images from orthomosaics according to satuelisa dataset.
# Saves crops as 76x76 squares (as opposed to 76x76 circles like satuelisa)

import os
from re import A
import shutil
import cv2

def read_data(filename, annotations=None):
    '''
    Read annotation data and return dictionary
    Input: filename
    Output: dictionary (key = color, values = coordinates)
    '''
    # dictionary to return
    if not annotations:
        annotations = {}

    # open file and read lines
    with open(filename, "r") as f:
        lines = f.readlines()

    # add details to dict
    for line in lines:
        # split details
        line = line.strip().split()
        color = line[1]
        coord = [int(line[2]),int(line[3])]

        # init new keys to []
        if color not in annotations.keys():
            annotations[color] = []

        # add to dict
        annotations[color].append(coord)
    
    # return
    return annotations
    
def crop_images(annotations, image_path, patch_size):
    '''
    Extract individual trees from image based on annotiation info in data
    Input: annotations (dict) and image_path , and patch size (approx size of trees in image)
    Output: img_crops dictionary (key = color, value = image) 
    '''
    # load image and convert to numpy
    img = cv2.imread(image_path)

    # crop images and append to img_crops (same keys as data)
    img_crops = dict.fromkeys(annotations.keys(), [])
    for color,coords in annotations.items(): 
        patches = []
        for coord in coords:
            patch = img[coord[1]-patch_size:coord[1]+patch_size, coord[0]-patch_size:coord[0]+patch_size]
            patches.append(patch)
        img_crops[color] = patches

    # return img_crops
    return img_crops

def save_image_crops(save_path, img_crops, date, overwrite=False, save_patch_size=76):
    '''
    Save cropped patches into appropriate folders based on color
    Input: save path and image crops (dict). Optional: overwrite save dir? and img save size.
    Output: None
    '''
    # loop over all colors
    for color,patches in img_crops.items():
        # create directory if not exists
        save_dir = f"{save_path}/{color}"
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        
        # if overwrite, delete directory (if exists) and make new one.
        if overwrite and os.path.isdir(save_dir):
            shutil.rmtree(save_dir)
            os.makedirs(save_dir)

        # determine counter (starts from 1)
        counter = len(os.listdir(save_dir)) + 1

        # loop over all patches, convert from numpy, and save 
        for patch in patches:
            filename = f"{save_dir}/{date}_{color}_{counter}.png"
            counter += 1
            resized_patch = cv2.resize(patch,[save_patch_size,save_patch_size])
            cv2.imwrite(filename, resized_patch)

def get_all_annotations(dates):
    ''' 
    Helper function for arranging annotations in dictionary
    Input: None
    Returns: dictionary [date][label] = [all images] for all dates 
    '''
    dates_dict = {}
    for date in dates:
        annotations = read_data(f'bb_repo/annotations/{date}.raw')
        dates_dict[date] = annotations
    return dates_dict

def count_dataset(dates):
    '''
    Function to count number of samples in each split of the given dates
    Input: dates indicating which flight
    Output: None (prints)
    '''

    dates_dict = get_all_annotations()

    labels = ['green', 'yellow', 'red', 'leafless']
    for date in dates:
        print (f"\nDate: {date}")
        for label in labels:
            print(f"{label} => {len(dates_dict[date][label])}")

        # split into train-test-val TODO: balanced labels implement
        train_percent = 85
        val_percent = 5
        total = sum([len(dates_dict[date][label]) for label in labels])
        num_train = train_percent * sum([len(dates_dict[date][label]) for label in labels]) // 100
        num_val = max(1,val_percent * sum([len(dates_dict[date][label]) for label in labels]) //100)        

        print (f"Train => {num_train}")
        print (f"Val => {num_val}")
        print (f"Test => {total-num_train-num_val}")


def crop_all_data():
    '''
    Function to crop all data and split according to classes for all flights.
    Input: None, 
    Output: None (saves to directory)
    '''
    scales = [60,50,50,50,50]
    flights = ['jun60', 'jul90', 'jul100', 'aug90', 'aug100']
    overwrite = True

    for scale, flight in zip(scales, flights):
        annotations = read_data(f'bb_repo/annotations/{flight}.raw')
        img_crops = crop_images(annotations, f'datasets/mosaics/{flight}.png', scale)
        save_image_crops(f'datasets/cropped_squares', img_crops, flight, overwrite=overwrite)
        overwrite = False


if __name__ == "__main__":
    # # run cropping for all data
    # scales = [60, 50, 50, 50, 50]
    dates = ['jun60','jul90','jul100','aug90','aug100']

    # overwrite = True
    # for scale,date in zip(scales, dates):
    #     annotations = read_data(f'bb_repo/annotations/{date}.raw')
    #     img_crops = crop_images(annotations, f'datasets/mosaics/{date}.png', scale)
    #     save_image_crops('datasets/cropped_squares', img_crops, date, overwrite=overwrite)
    #     overwrite = False 
    # 

    count_dataset(dates)


