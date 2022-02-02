# ------------------------------------------------------------ #
#
# file : predict.py
# author : Chuanbo
# Predict on the testing images with the trained model 
# 
# ------------------------------------------------------------ #

import os
from data import *
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import backend as keras
from utils.io.write import npToNiiAffine
from utils.learning.losses import dice_loss
from utils.learning.metrics import f1, sensitivity, specificity, precision
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from utils.io.read import readDatasetPart, reshapeDataset, getAffine, getAffine_subdir
 
current_path = str(os.path.dirname(os.path.realpath(__file__)))

# training images, labels, testing images, labels
x_train ,y_train, x_test, y_test = load_data("./data/")
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# model
size_x = 36
size_y = 256
size_z = 256

unet_input = Input(shape=(size_x, size_y, size_z, 1))

conv_1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                    input_shape=(size_x, size_y, size_z, 1))(unet_input)
conv_1 = Dropout(0.2)(conv_1)
conv_1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv_1)
pool_1 = MaxPooling3D((2, 2, 2))(conv_1)

#
conv_2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool_1)
conv_2 = Dropout(0.2)(conv_2)
conv_2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv_2)
pool_2 = MaxPooling3D((2, 2, 2))(conv_2)

#
conv_3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool_2)
conv_3 = Dropout(0.2)(conv_3)
conv_3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv_3)

#
up_1 = UpSampling3D(size=(2, 2, 2))(conv_3)
up_1 = concatenate([conv_2, up_1], axis=4)
conv_4 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(up_1)
conv_4 = Dropout(0.2)(conv_4)
conv_4 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv_4)

#
up_2 = UpSampling3D(size=(2, 2, 2))(conv_4)
up_2 = concatenate([conv_1, up_2], axis=4)
conv_5 = Conv3D(32, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(up_2)
conv_5 = Dropout(0.2)(conv_5)
conv_5 = Conv3D(32, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv_5)

#
conv_6 = Conv3D(2, (1, 1, 1), activation='relu', padding='same', kernel_initializer='he_normal')(conv_5)
conv_6 = BatchNormalization(axis=4)(conv_6)
conv_7 = Conv3D(1, (1, 1, 1), activation='sigmoid', padding='same', kernel_initializer='he_normal')(conv_6)

model = Model(inputs=unet_input, outputs=conv_7)

model = load_model('simple_3d_unet_small_volume_3modalities_aug.hdf5',custom_objects={'dice_loss': dice_loss
                                                              ,'sensitivity': sensitivity
                                                              ,'specificity': specificity
                                                              ,'precision': precision})
# training settings
model.compile(optimizer=Adam(lr=1e-5),loss=dice_loss,metrics=[sensitivity, specificity, precision])
          
#model_checkpoint = ModelCheckpoint('temp/simple_3d_unet.hdf5',verbose=1, save_best_only=True)
#model_checkpoint,          
#model.fit(x_train,y_train,batch_size=1,epochs=10,validation_split=0.2,verbose=1
#          ,callbacks=[model_checkpoint])
#model.save('simple_3d_unet.hdf5')

# predict
prediction = model.predict(x_test, verbose=1)
for count in range(prediction.shape[0]):
    npToNiiAffine(prediction[count], getAffine_subdir("./data/test/images/"), "./data/prediction/"+(str(count + 1).zfill(2) + ".nii.gz"))



