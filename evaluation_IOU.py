# ------------------------------------------------------------ #
#
# file : evaluation_IOU.py
# author : Chuanbo
# Evaluate the IOU and Dice score from predictions and ground truth
# 
# ------------------------------------------------------------ #
import numpy as np
import nibabel as nib
import cv2
import os
import sys
from data import load_data,save_results

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
    #print("image file list: ", file_list)
    image = nib.load(os.path.join(path, file_list[0]))
    arr = np.empty(( image.shape[0], image.shape[1], image.shape[2], len(file_list))).astype(np.float32)
    del image
    count = 0
    for filename in file_list:
        img = nib.load(os.path.join(path, filename)).get_data()[:,:,:,0]
        arr[:, :, :, count] = img
        count += 1
        if(count>=len(file_list)):
            break


    arr = np.array(arr)
    #x_train = np.reshape(x_train,(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
    return arr


def read_nii2(path):
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
    #print("image file list: ", file_list)
    image = nib.load(os.path.join(path, file_list[0]))
    arr = np.empty(( image.shape[0], image.shape[1], image.shape[2], len(file_list))).astype(np.float32)
    del image
    count = 0
    for filename in file_list:
        img = nib.load(os.path.join(path, filename)).get_data()
        arr[:, :, :, count] = img
        count += 1
        if(count>=len(file_list)):
            break


    arr = np.array(arr)
    #x_train = np.reshape(x_train,(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
    return arr

prediction_path = "./data/prediction/simple_3d_unet_small_volume_4modalities_NOaugV2_15000/"
y_test = read_nii2("./data/prediction/y_test/")
arr = read_nii(prediction_path)

#print(y_test.shape)
#print(arr.shape)
#print(np.amax(y_test), np.amin(y_test))
#print(np.amax(arr), np.amin(arr))
#save_results(y_test,"./y_test.nii")
#save_results(arr,"./prediction.nii")

xdim = y_test.shape[0]
ydim = y_test.shape[1]
zdim = y_test.shape[2]
channels = y_test.shape[3]
'''
# visualize results ###################################################
for channel in range(0,17):
    visualization = np.zeros((xdim, ydim, zdim, 3))
    for x in range(xdim):
        for y in range(ydim):
            for z in range(zdim):
                if y_test[x, y, z, channel]>0.2 and arr[x, y, z, channel]>0.2:
                    visualization[x,y,z,0] = 1
                if y_test[x, y, z, channel]>0.2 and arr[x, y, z, channel]<0.2:
                    visualization[x,y,z,1] = 1
                if y_test[x, y, z, channel]<0.2 and arr[x, y, z, channel]>0.2:
                    visualization[x,y,z,2] = 1
    save_results(visualization,"./visualization/visualization"+(str)(channel)+".nii")
#########################################################################
'''
# compute IOU and Dice ###################################################
false_positives = 0
false_negtives = 0
true_positives = 0
for channel in range(0,9):
    for x in range(xdim):
        for y in range(ydim):
            for z in range(zdim):
                if y_test[x, y, z, channel]>0.2 and arr[x, y, z, channel]>0.2:
                    true_positives += 1
                if y_test[x, y, z, channel]>0.2 and arr[x, y, z, channel]<0.2:
                    false_negtives += 1
                if y_test[x, y, z, channel]<0.2 and arr[x, y, z, channel]>0.2:
                    false_positives += 1

IOU = float(true_positives) / (true_positives + false_negtives + false_positives)
Dice1 = 2*float(true_positives) / (2*true_positives + false_negtives + false_positives)

false_positives = 0
false_negtives = 0
true_positives = 0
for channel in range(10,17):
    for x in range(xdim):
        for y in range(ydim):
            for z in range(zdim):
                if y_test[x, y, z, channel]>0.2 and arr[x, y, z, channel]>0.2:
                    true_positives += 1
                if y_test[x, y, z, channel]>0.2 and arr[x, y, z, channel]<0.2:
                    false_negtives += 1
                if y_test[x, y, z, channel]<0.2 and arr[x, y, z, channel]>0.2:
                    false_positives += 1

IOU = float(true_positives) / (true_positives + false_negtives + false_positives)
Dice2 = 2*float(true_positives) / (2*true_positives + false_negtives + false_positives)

print("weight file: " + prediction_path.split("/")[-2])
print("IOU = " + str(IOU))
print("Mean Dice = " + str(np.mean([Dice1*100, Dice2*100])))
print("Dice SD = " + str(np.std([Dice1*100, Dice2*100])))
#########################################################################

