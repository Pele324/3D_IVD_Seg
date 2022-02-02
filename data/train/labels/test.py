import numpy as np
import nibabel as nib
import cv2
import os
import sys
import shutil
path="."
highest_level_subdir_list = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
highest_level_subdir_list.sort()

for subdir in highest_level_subdir_list:
    path2 = path+"/"+subdir
    for dirName, subdirList, fileList in os.walk(path2):
        for filename in fileList:
            if ".nii" in filename.lower():
                realfilename, file_extension = os.path.splitext(filename)
                new_name1 = realfilename+"1"+file_extension
                new_name2 = realfilename+"2"+file_extension
                shutil.copy(path2+"/"+filename, path2+"/"+new_name1)
                shutil.copy(path2+"/"+filename, path2+"/"+new_name2)
