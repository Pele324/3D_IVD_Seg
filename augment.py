# ------------------------------------------------------------ #
#
# file : augment.py
# author : Chuanbo
# Augmentation functions including elastic transformation
# 
# ------------------------------------------------------------ #

import SimpleITK as sitk
import scipy.ndimage
from dltk.io.augmentation import elastic_transform
from dltk.io.preprocessing import *


def augment(images, label):
    # Normalise volume images
    for i in range(len(images)):
        images[i] = whitening(images[i])

    label = whitening(label)

    # Generate random numbers
    np.random.seed()
    category = np.random.choice(3, 1)

    # category = 1:nothing; 2:affine_transform; 3:distortion
    if category == 0:
        return images, label
    elif category == 1:
        # apply random number of affine operations on (images, label)
        numTrans = np.random.randint(1, 5, size=1)
        allowedTrans = [0, 1, 2, 3]
        whichTrans = np.random.choice(allowedTrans, numTrans, replace=False)

        # Rotation
        if 0 in whichTrans:
            axes = np.random.choice([0, 1, 2], 2, replace=False)
            theta = float(np.around(np.random.uniform(-30.0, 30.0, size=1), 2))
            for i in range(len(images)):
                images[i] = rotate(images[i], theta, axes)
            label = rotate(label, theta, axes, isseg=True)

        # Scaling
        if 1 in whichTrans:
            scalefactor = float(np.around(np.random.uniform(0.7, 1.3, size=1), 2))
            for i in range(len(images)):
                images[i] = scale(images[i], scalefactor)
            label = scale(label, scalefactor, isseg=True)

        # Flip
        if 2 in whichTrans:
            axes = np.random.choice(3, 1)
            for i in range(len(images)):
                images[i] = np.flip(images[i], axes)
            label = np.flip(label, axes)

        # Translate
        if 3 in whichTrans:
            offset = list(np.random.randint(-20, 20, size=2))
            offset.append(0)
            for i in range(len(images)):
                images[i] = translate(images[i], offset)
            label = translate(label, offset, isseg=True)

        return images, label
    else:
        return distortion1(images, label)


def augment2(images, label):
    # apply random number of affine operations on (images, label)
    numTrans = np.random.randint(1, 8, size=1)
    allowedTrans = [0, 1, 2, 3, 4, 5, 6]
    whichTrans = np.random.choice(allowedTrans, numTrans, replace=False)

    # print(whichTrans)

    # Rotation
    if 0 in whichTrans:
        axes = (0, 1)
        theta = float(np.around(np.random.uniform(-30.0, 30.0, size=1), 2))
        for i in range(len(images)):
            images[i] = rotate(images[i], theta, axes)
        label = rotate(label, theta, axes, isseg=True)

    # Scaling
    if 1 in whichTrans:
        scalefactor = float(np.around(np.random.uniform(0.7, 1.3, size=1), 2))
        for i in range(len(images)):
            images[i] = scale(images[i], scalefactor)
        label = scale(label, scalefactor, isseg=True)

    # Flip
    if 2 in whichTrans:
        axes = np.random.choice(3, 1)
        for i in range(len(images)):
            images[i] = np.flip(images[i], axes)
        label = np.flip(label, axes)

    # Translate
    if 3 in whichTrans:
        offset = list(np.random.randint(-20, 20, size=2))
        offset.append(0)
        for i in range(len(images)):
            images[i] = translate(images[i], offset)
        label = translate(label, offset, isseg=True)

    # Distortion
    if any(i >= 4 for i in whichTrans):
        imagse, label = distortion1(images, label)

    return images, label


def distortion1(images, label):
    temp = images.copy()
    temp.append(label)
    image = np.stack(temp, axis=-1).astype(np.float32)
    # alpha = list(np.random.randint(1e3, 1e5, size=3))
    # sigma = list(np.random.randint(10, 25, size=3))

    # image = elastic_transform(image, alpha=[1e4, 1e5 / 2, 1e4], sigma=[25,25,25])  # [z, y, x]
    image = elastic_transform(image, alpha=[3e5, 5e5, 1e5], sigma=[25, 25, 25])  # [z, y, x]

    shape = label.shape

    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                for m in range(len(images)):
                    images[m][i][j][k] = image[i][j][k][m]
                label[i][j][k] = image[i][j][k][-1]

    return images, label


def distortion2(images, label):
    num_controlpoints = 4
    std_deformation_sigma = 15
    spatial_rank = 3

    temp = images.copy()
    temp.append(label)
    image = np.stack(temp, axis=-1).astype(np.float32)

    squeezed_shape = image.shape[:spatial_rank]
    itkimg = sitk.GetImageFromArray(np.zeros(squeezed_shape))
    trans_from_domain_mesh_size = [num_controlpoints] * itkimg.GetDimension()
    bspline_transformation = sitk.BSplineTransformInitializer(itkimg, trans_from_domain_mesh_size)

    params = bspline_transformation.GetParameters()
    params_numpy = np.asarray(params, dtype=float)
    params_numpy = params_numpy + np.random.randn(params_numpy.shape[0]) * std_deformation_sigma
    params = tuple(params_numpy)
    bspline_transformation.SetParameters(params)

    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkBSpline)

    squeezed_image = np.squeeze(image)
    while squeezed_image.ndim < spatial_rank:
        # pad to the required number of dimensions
        squeezed_image = squeezed_image[..., None]
    sitk_image = sitk.GetImageFromArray(squeezed_image)

    resampler.SetReferenceImage(sitk_image)
    resampler.SetDefaultPixelValue(0)

    resampler.SetTransform(bspline_transformation)
    out_img_sitk = resampler.Execute(sitk_image)
    out_img = sitk.GetArrayFromImage(out_img_sitk)

    shape = label.shape

    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                for m in range(len(images)):
                    images[m][i][j][k] = out_img[i][j][k][m]
                label[i][j][k] = out_img[i][j][k][-1]

    return images, label


def translate(image, offset, isseg=False):
    order = 0 if isseg is True else 5
    return scipy.ndimage.interpolation.shift(image, (int(offset[0]), int(offset[1]), int(offset[2])), order=order)


def scale(image, factor, isseg=False):
    order = 0 if isseg is True else 5

    height, width, depth = image.shape
    zheight = int(np.round(factor * height))
    zwidth = int(np.round(factor * width))
    zdepth = int(np.round(factor * depth))

    if factor < 1.0:
        newimg = np.zeros_like(image)
        row = (height - zheight) // 2
        col = (width - zwidth) // 2
        layer = (depth - zdepth) // 2
        newimg[row:row + zheight, col:col + zwidth, layer:layer + zdepth] = scipy.ndimage.interpolation.zoom(image, (
        float(factor), float(factor), float(factor)), order=order)[0:zheight, 0:zwidth, 0:zdepth]

        return newimg
    elif factor > 1.0:
        row = (zheight - height) // 2
        col = (zwidth - width) // 2
        layer = (zdepth - depth) // 2

        newimg = scipy.ndimage.interpolation.zoom(image[row:row + zheight, col:col + zwidth, layer:layer + zdepth],
                                    (float(factor), float(factor), float(factor)), order=order)

        extrah = (newimg.shape[0] - height) // 2
        extraw = (newimg.shape[1] - width) // 2
        extrad = (newimg.shape[2] - depth) // 2
        newimg = newimg[extrah:extrah + height, extraw:extraw + width, extrad:extrad + depth]

        return newimg
    else:
        return image


def rotate(image, theta, axes, isseg=False):
    order = 0 if isseg is True else 5
    return scipy.ndimage.interpolation.rotate(image, float(theta), axes=axes, reshape=False, order=order)