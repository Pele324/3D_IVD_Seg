# ------------------------------------------------------------ #
#
# file : visualization.py
# author : Chuanbo
# Visualize the results
# 
# ------------------------------------------------------------ #

import numpy as np
import nibabel as nib
import cv2
import os
import sys
import SimpleITK as sitk
from utils.io.write import npToNiiAffine
from utils.io.read import readDatasetPart, reshapeDataset, getAffine, getAffine_subdir

def contour(arr):
    out = np.copy(arr)
    for x in range(1,arr.shape[0]-1):
        for y in range(1,arr.shape[1]-1):
            for z in range(1,arr.shape[2]-1):
                #just do 2d contour
                if arr[x,y,z] == 1 and arr[x,y+1,z] == 1 and arr[x,y-1,z] == 1 and arr[x-1,y,z] == 1 and arr[x-1,y+1,z] == 1 and arr[x-1,y-1,z] == 1 and arr[x+1,y,z] == 1 and arr[x+1,y+1,z] == 1 and arr[x+1,y-1,z] == 1:
                    out[x,y,z] = 0
    return out
                                
                        
                    
                                    


names = ["G1","H1"]
for name in names:
    #load nii files
    image = nib.load("data/temp_visualization/" + name + "_opp.nii").get_fdata()
    label = nib.load("data/temp_visualization/" + name + "-Labels.nii").get_fdata()
    prediction = nib.load("data/temp_visualization/" + name + "_predict_3modalities4.nii.gz").get_fdata()
    
    #convert to numpy arrays
    image = np.array(image)
    label = np.array(label)
    prediction = np.array(prediction)
    
    #conbine 3 channels in the prediction into a single channel mask
    x_dim = prediction.shape[0]
    y_dim = prediction.shape[1]
    z_dim = prediction.shape[2]
    num_channels = prediction.shape[3]
    prediction2 = np.zeros((x_dim,y_dim,z_dim), dtype = prediction.dtype)
    for x in range(x_dim):
        for y in range(y_dim):
            for z in range(z_dim):
                for channel in range(num_channels):
                    if prediction[x,y,z,channel] > 0.02:
                        prediction2[x,y,z] = 1
                    else:
                        prediction2[x,y,z] = 0
    prediction = prediction2
    
    #binarize the label as well
    label2 = np.zeros((x_dim,y_dim,z_dim), dtype = label.dtype)
    for x in range(x_dim):
        for y in range(y_dim):
            for z in range(z_dim):
                if label[x,y,z] > 0.02:
                    label2[x,y,z] = 1
                else:
                    label2[x,y,z] = 0
    label = label2
    
    #find contour of the label and the prediction
    label_contour = contour(label)
    prediction_contour = contour(prediction)
    npToNiiAffine(prediction_contour, getAffine_subdir("./data/test/images/"), "data/temp_visualization/prediction_contour.nii")
    npToNiiAffine(label_contour, getAffine_subdir("./data/test/images/"), "data/temp_visualization/label_contour.nii")
