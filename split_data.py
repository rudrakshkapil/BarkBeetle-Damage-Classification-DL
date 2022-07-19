## Script to organize data, split it across train-val-test, and generate corresponding csv files.
# $$ Alternative: use the .raw files to split -- if loading different images is too slow 

import numpy as np
import os
import cv2
import shutil
import csv

def overwrite_directory(path_to_dir):
    '''
    Overwrite an existing directory with a new one of same name
    If none exists, creates new one.
    Input: path to directory
    Output: None
    '''
    # if exists, delete. Either way create new one
    if os.path.isdir(path_to_dir):
        shutil.rmtree(path_to_dir)
    os.makedirs(path_to_dir)

def copy_images_to_directory(img_names, src_dir, dst_dir):
    '''
    Copy the images specified in img_names from source to destination
    Input: list of image names, source directory path, destination directory path
    Output: None 
    '''
    # loop over images
    for name in img_names:
        # make src and dst names
        src = f"{src_dir}/{name}"
        dst = f"{dst_dir}/{name}"

        # copy
        shutil.copyfile(src,dst)

def generate_csv_file(img_names, split_dir, label):
    '''
    Generates csv file for the current split in required format:
        name.png, xmin, ymin, xmax, ymax, label
    Input: list of img names, dir for current split, label 
    Output: Filename (also saves file to split_dir)
    '''
    # check if file exists -- split name is extracted as last part of split_dir
    split_name = split_dir.split("/")[-1]  
    csv_file_path = f"{split_dir}/_{split_name}_info.csv"
    csv_exists = os.path.isfile(csv_file_path)

    # create (open file) in read+append mode and create csv writer
    csv_file = open(csv_file_path, mode='a+', newline='')
    writer = csv.writer(csv_file)

    # add header if doesn't exist
    if not csv_exists:
        writer.writerow(['image_path', 'xmin', 'ymin', 'xmax', 'ymax', 'label'])

    # get image size (entire image is bounding box)
    img = cv2.imread(f"{split_dir}/{img_names[0]}")
    xmin, ymin = 0, 0
    xmax, ymax = img.shape[0], img.shape[1]

    # loop over all images
    for name in img_names:
        # generate and write line
        #line = f"{name:23}{xmin:4},{ymin:4},{xmax:4},{ymax:4}, {label}"
        line = [name, xmin, ymin, xmax, ymax, label]
        writer.writerow(line)
    
    # close file
    csv_file.close()

    # return filename
    return csv_file_path

def shuffle_csv_lines(csv_file):
    ''' 
    Shuffles lines (except first) in csv file
    Input: csv_file
    Output: None (in place shuffling)
    '''
    with open(csv_file, 'r') as f:
        lines = f.readlines()
        header = lines[0]
        body = lines[1:]
    np.random.shuffle(body)
    with open(csv_file, 'w') as f:
        f.writelines(header)
        f.writelines(body)



def split_data(split_params=None):
    '''
    Read image names, split into train-val-test folders, anf generate corresponding csv 
    (random heights)
    Input: (opt) split_params dictionary for different options
    Output: dict containing X_train, y_train, X_test, and y_test
    '''
    # optional param
    if not split_params:
        split_params = {}

    # extract options
    train_percent = split_params.get('train_percent', 85)
    val_percent = split_params.get('val_percent', 5)        # test = 100 - train - val
    balanced_labels = split_params.get('balanced', False)
    include_ground = split_params.get('include_ground', False)
    crop_dir = split_params.get('crop_dir', 'datasets/cropped_squares')
    split_dir = split_params.get('split_dir', 'datasets/split')

    # determine labels
    labels = ['green', 'red', 'yellow', 'leafless']
    if include_ground: 
        labels.append("ground")

    # create directories to store splits
    train_dir = f"{split_dir}/train"
    val_dir = f"{split_dir}/val"
    test_dir = f"{split_dir}/test"
    
    overwrite_directory(train_dir)
    overwrite_directory(val_dir)
    overwrite_directory(test_dir)

    # loop over labels
    for label in labels:
        # go through directory and get names of images
        label_crop_dir = f"{crop_dir}/{label}"
        image_names = os.listdir(label_crop_dir)

        # shuffle (for heights)
        np.random.shuffle(image_names)
        
        # split into train-test-val TODO: balanced labels implement
        num_train = train_percent * len(image_names) // 100
        num_val = max(1,val_percent * len(image_names)//100)        

        train_names = image_names[:num_train]
        val_names = image_names[num_train:num_train+num_val]
        test_names = image_names[num_train+num_val:]

        # copy images to new directories
        copy_images_to_directory(train_names, label_crop_dir, train_dir)
        copy_images_to_directory(val_names, label_crop_dir, val_dir)
        copy_images_to_directory(test_names, label_crop_dir, test_dir)

        # generate csv files (format: name.png, xmin, ymin, xmax, ymax, label)
        train_csv_file = generate_csv_file(train_names, train_dir, label)
        val_csv_file = generate_csv_file(val_names, val_dir, label)
        test_csv_file = generate_csv_file(test_names, test_dir, label)

    # shuffle lines
    shuffle_csv_lines(train_csv_file)
    shuffle_csv_lines(val_csv_file)
    shuffle_csv_lines(test_csv_file)


def generate_csv_file_for_mosaics(all_lines, split_dir, split_name):
    '''
    Generates csv file for the current split in required format:
        name.png, xmin, ymin, xmax, ymax, label
    Input: lines to print to file, dir for current split
    Output: Filename (also saves file to split_dir)
    '''
    # check if file exists -- split name is extracted as last part of split_dir
    csv_file_path = f"{split_dir}/_{split_name}_info.csv"

    # create (open file) in read+append mode and create csv writer
    csv_file = open(csv_file_path, mode='w', newline='')
    writer = csv.writer(csv_file)

    # output to file
    writer.writerow(['image_path', 'xmin', 'ymin', 'xmax', 'ymax', 'label'])
    csv_file.writelines(all_lines)
    
    # close file
    csv_file.close()

    # return filename
    return csv_file_path


def split_data_orthomosaics(split_params):
    '''
    Split data and generate CSVs for each split. 
    Input: parameters for current split
    Output: None.
    '''
    train_percent = split_params.get('train_percent', 85)
    val_percent = split_params.get('val_percent', 5)        # test = 100 - train - val
    scales = split_params.get('scales', [60, 50, 50, 50, 50])
    dates = split_params.get('dates', ['jun60','jul90','jul100','aug90','aug100'])
    include_ground = split_params.get('include_ground', True)
    annotations_dir = split_params.get('annotations_dir', './datasets/annotations')
    mosaic_dir = split_params.get('mosaic_dir', './datasets/mosaics')

    # combine all raw info together
    all_lines = []
    for scale,date in zip(scales,dates):
        # open and read date.raw file
        raw_file = open(f"{annotations_dir}/{date}.raw")
        lines = raw_file.readlines()

        # fix each line for current date
        date_lines = []
        for line in lines:
            line = line.strip().split()

            # get name and label
            img_name = f"{line[0]}.png"
            label = line[1]

            # skip if ground not needed
            if label == 'ground' and not include_ground:
                continue

            # get extreme coordinates (xmin,xmax,ymin,ymax) from center (x,y)
            x, y = int(line[2]), int(line[3])
            xmin = x-scale
            xmax = x+scale
            ymin = y-scale
            ymax = y+scale

            # create and append line to date_lines 
            date_lines.append(f"{img_name},{xmin},{ymin},{xmax},{ymax},{label}\n")

        # add to all lines and close current file
        all_lines.extend(date_lines)
        raw_file.close()

    # shuffle all lines
    np.random.shuffle(all_lines)

    # split all into train-test-val TODO: balanced labels implement
    num_train = train_percent * len(all_lines) // 100
    num_val = val_percent * len(all_lines)     // 100

    train_lines = all_lines[:num_train]
    val_lines = all_lines[num_train:num_train+num_val]
    test_lines = all_lines[num_train+num_val:]

    # generate csv files (format: name.png, xmin, ymin, xmax, ymax, label)
    generate_csv_file_for_mosaics(train_lines, mosaic_dir, "train")
    generate_csv_file_for_mosaics(val_lines, mosaic_dir, "val")
    generate_csv_file_for_mosaics(test_lines, mosaic_dir, "test")







if __name__ == "__main__":
    #split_data_orthomosaics({})
    pass
