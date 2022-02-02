# ------------------------------------------------------------ #
#
# file : data.py
# author : Chuanbo
# dataloader
# 
# ------------------------------------------------------------ #

import numpy as np
import nibabel as nib
import cv2
import os
import sys

def load_data(path):
    #print("test")
    path_train_images = path + "train/images/"
    path_train_labels = path + "train/labels/"
    path_test_images = path + "test/images/"
    path_test_labels = path + "test/labels/"

    #path_list = [path_train_images]
    x_train = read_modality_nii(path_train_images)
    y_train = read_modality_nii(path_train_labels)
    x_test = read_modality_nii(path_test_images)
    y_test = read_modality_nii(path_test_labels)
    x_train = x_train / np.amax(x_train)
    y_train = y_train / np.amax(y_train)
    x_test = x_test / np.amax(x_test)
    y_test = y_test / np.amax(y_test)
    #print(np.amax(x_train),np.amin(x_train),np.mean(x_train),np.median(x_train))
    #print(np.amax(y_train),np.amin(y_train),np.mean(y_train),np.median(y_train))
    #print(np.amax(x_test),np.amin(x_test),np.mean(x_test),np.median(x_test))
    #print(np.amax(y_test),np.amin(y_test),np.mean(y_test),np.median(y_test))
    
    ### returning the np arrays ###################
    return x_train, y_train, x_test, y_test


#read multi modality MRI data into a np array
def read_modality_nii(path): # example: path="./data/train/images/"
    #get file list
    highest_level_subdir_list = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
    highest_level_subdir_list.sort()
    #print("sub dir list: ", highest_level_subdir_list)
    temp = read_nii(path+highest_level_subdir_list[0]+'/')
    

    arr = np.empty((len(highest_level_subdir_list), temp.shape[0], temp.shape[1], temp.shape[2], temp.shape[3])).astype(np.float32)
    
    count = 0
    for subdir in highest_level_subdir_list:
        arr[count,:, :, :, :] = read_nii(path+subdir+'/')
        count += 1
    return arr
        
def read_nii(path):
    file_list = []
    for FileNameLength in range(0,100):
        for dirName, subdirList, fileList in os.walk(path):
            for filename in fileList:
                # check file extension
                if ".nii" in filename.lower() and len(filename) == FileNameLength:
                    file_list.append(filename)
            break
    arr = []
    file_list.sort()
    #file_list = [filename for filename in file_list if "fat" not in filename]
    file_list = [filename for filename in file_list if "inn" not in filename]
    #print("image file list: ", file_list)
    image = nib.load(os.path.join(path, file_list[0]))
    zero_padded_x = image.shape[0]
    zero_padded_y = image.shape[1]
    zero_padded_z = image.shape[2]
    while zero_padded_x % 4 != 0:
        zero_padded_x += 1
    while zero_padded_y % 4 != 0:
        zero_padded_y += 1
    while zero_padded_z % 4 != 0:
        zero_padded_z += 1
    arr = np.zeros(( zero_padded_x, zero_padded_y, zero_padded_z, len(file_list))).astype(np.float32)
    del image
    count = 0
    for filename in file_list:
        temp = nib.load(os.path.join(path, filename)).get_data()
        for x in range(temp.shape[0]):
            for y in range(temp.shape[1]):
                for z in range(temp.shape[2]):
                    arr[x, y, z, count] = temp[x,y,z]
        count += 1
        if(count>=len(file_list)):
            break


    arr = np.array(arr)
    #x_train = np.reshape(x_train,(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
    return arr






def save_results(np_array, outpath):
    axes = np.eye(4)
    axes[0][0] = -1
    axes[1][1] = -1
    image = nib.Nifti1Image(np_array, axes)
    nib.save(image, outpath)
