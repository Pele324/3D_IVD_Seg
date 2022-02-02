from augment import *

name = 'A1'

# Read images using simpleitk
image_fat = sitk.GetArrayFromImage(sitk.ReadImage(name + '_fat.nii'))
image_inn = sitk.GetArrayFromImage(sitk.ReadImage(name + '_inn.nii'))
image_opp = sitk.GetArrayFromImage(sitk.ReadImage(name + '_opp.nii'))
image_wat = sitk.GetArrayFromImage(sitk.ReadImage(name + '_wat.nii'))
image_label = sitk.GetArrayFromImage(sitk.ReadImage(name + '-Labels.nii'))

images = [image_fat, image_inn, image_opp, image_wat]

# images, image_label = augment(images, image_label)

# image_label = translate(image_label, [20, 0, 0])
# image_label = scale(image_label, 0.3)
# image_label = rotate(image_label, 45, (0, 1, 0))
# image_label = flip(image_label, (0, 1))
images, image_label = distortion1(images, image_label)
# images, image_label = distortion2(images, image_label)

image_fat = images[0]
image_inn = images[1]
image_opp = images[2]
image_wat = images[3]

print(np.max(image_label))

# Write images using simpleitk
sitk.WriteImage(sitk.GetImageFromArray(image_fat), name + '_fat_aug.nii')
sitk.WriteImage(sitk.GetImageFromArray(image_inn), name + '_inn_aug.nii')
sitk.WriteImage(sitk.GetImageFromArray(image_opp), name + '_opp_aug.nii')
sitk.WriteImage(sitk.GetImageFromArray(image_wat), name + '_wat_aug.nii')

original = sitk.ReadImage(name + '-Labels.nii')
label = sitk.GetImageFromArray(image_label)
label.SetOrigin(original.GetOrigin())
label.SetSpacing(original.GetSpacing())
label.SetDirection(original.GetDirection())

sitk.WriteImage(label, name + '_label_aug.nii')

