# ------------------------------------------------------------ #
#
# author : Chuanbo
# Find the centers of each IVD and split large nii into small 
# 36x36x36 nii volumes 
# ------------------------------------------------------------ #

import numpy as np
import nibabel as nib
from scipy.ndimage.measurements import label
from utils.io.write import npToNiiAffine
from utils.io.read import readDatasetPart, reshapeDataset, getAffine, getAffine_subdir

patient = "H1"
label_path = "data/prediction/02_3modalities4.nii.gz"
#label_path = "data/test/labels/"+patient+"/"+patient+"-Labels.nii"
opp_path = "data/test/images/"+patient+"/"+patient+"_opp.nii"
fat_path = "data/test/images/"+patient+"/"+patient+"_fat.nii"
wat_path = "data/test/images/"+patient+"/"+patient+"_wat.nii"
wat_path = "data/test/images/"+patient+"/"+patient+"_inn.nii"

label_arr = nib.load(label_path)
label_arr = label_arr.get_fdata() ### (36, 256, 256), float64, np array

opp_arr = nib.load(opp_path)
opp_arr = opp_arr.get_fdata() ### (36, 256, 256), float64, np array

fat_arr = nib.load(fat_path)
fat_arr = fat_arr.get_fdata() ### (36, 256, 256), float64, np array

wat_arr = nib.load(wat_path)
wat_arr = wat_arr.get_fdata() ### (36, 256, 256), float64, np array

inn_arr = nib.load(wat_path)
inn_arr = inn_arr.get_fdata() ### (36, 256, 256), float64, np array

label_arr[label_arr > 0.02] == 1
label_arr[label_arr <= 0.02] == 0

### get the single-channel segmentation mask from 3 channels
x_dim = label_arr.shape[0]
y_dim = label_arr.shape[1]
z_dim = label_arr.shape[2]

if len(label_arr.shape) == 4:
    num_channels = label_arr.shape[3]
    single_chennel_arr = np.zeros((x_dim,y_dim,z_dim), dtype = label_arr.dtype)

    for x in range(x_dim):
        for y in range(y_dim):
            for z in range(z_dim):
                for channel in range(num_channels):
                    if label_arr[x,y,z,channel] > 0.02:
                        single_chennel_arr[x,y,z] = 1
                    else:
                        single_chennel_arr[x,y,z] = 0
    label_arr = single_chennel_arr


### Get the center of each connected component
s = np.ones((3,3,3))

labeled, num_components = label(label_arr, structure = s)

unique, counts = np.unique(labeled, return_counts=True)
print(unique)
print(counts)
for label in unique:
    if counts[label] < 1000:
        labeled[labeled == label] = 0

unique, counts = np.unique(labeled, return_counts=True)

'''delete = np.array([0])
#unique = np.setdiff1d(labeled,delete)
unique = np.array([el for el in unique if el not in delete])
'''

print(unique)
print(counts)
    

centers = np.zeros((unique.size, 3))
for label in range(unique.size):
    print(label)
    voxel_count = 0
    for x in range(x_dim):
        for y in range(y_dim):
            for z in range(z_dim):
                if labeled[x,y,z] == label+1:
                    centers[label,0] += x
                    centers[label,1] += y
                    centers[label,2] += z
                    voxel_count += 1
    print(centers[label,:], voxel_count)
    centers[label,:] /= voxel_count
    
centers = centers[centers[:,2].argsort()] #sort the centers by their z values
centers = centers.astype(int)

### For each center:
### Generate a new .nii for the 35*35*25 volume around it
k = 0
for center in centers:
    print(center)
    label_volume = np.zeros((35,35,25))
    opp_volume = np.zeros((35,35,25))
    fat_volume = np.zeros((35,35,25))
    wat_volume = np.zeros((35,35,25))
    inn_volume = np.zeros((35,35,25))
    y_offset = center[1] - np.int(label_volume.shape[1]/2)
    z_offset = center[2] - np.int(label_volume.shape[2]/2)
    for x in range(label_volume.shape[0]):
        for y in range(label_volume.shape[1]):
            for z in range(label_volume.shape[2]):
                label_volume[x,y,z] = label_arr[x, y + y_offset, z + z_offset]
                opp_volume[x,y,z] = opp_arr[x, y + y_offset, z + z_offset]
                fat_volume[x,y,z] = fat_arr[x, y + y_offset, z + z_offset]
                wat_volume[x,y,z] = wat_arr[x, y + y_offset, z + z_offset]
                inn_volume[x,y,z] = inn_arr[x, y + y_offset, z + z_offset]
    k += 1
    npToNiiAffine(label_volume, getAffine_subdir("./data/test/images/"),
                  "new_dataset/"+patient+"_small_label_volume"+(str)(k)+".nii")
    npToNiiAffine(opp_volume, getAffine_subdir("./data/test/images/"),
                  "new_dataset/"+patient+"_small_opp_volume"+(str)(k)+".nii")
    npToNiiAffine(fat_volume, getAffine_subdir("./data/test/images/"),
                  "new_dataset/"+patient+"_small_fat_volume"+(str)(k)+".nii")
    npToNiiAffine(wat_volume, getAffine_subdir("./data/test/images/"),
                  "new_dataset/"+patient+"_small_wat_volume"+(str)(k)+".nii")
    npToNiiAffine(wat_volume, getAffine_subdir("./data/test/images/"),
                  "new_dataset/"+patient+"_small_inn_volume"+(str)(k)+".nii")
    
























                
    
    

