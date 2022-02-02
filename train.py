# ------------------------------------------------------------ #
#
# file : train.py
# author : Chuanbo
# Training
# 
# ------------------------------------------------------------ #

import os
from data import *
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import backend as keras
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, CSVLogger, ReduceLROnPlateau
from utils.learning.losses import dice_loss
from utils.learning.metrics import f1, sensitivity, specificity, precision
from sklearn.model_selection import train_test_split, KFold
from keras.preprocessing.image import ImageDataGenerator
from augment import *
 
# training images, labels, testing images, labels
x_train ,y_train, x_test, y_test = load_data("./data/")

print("train images: ", x_train.shape)
print("train labels: ", y_train.shape)
print("test images: ", x_test.shape)
print("test labels: ", y_test.shape)



### data augmentation
for num_augmentation in range(2):
    for num_volume in range(49):
        # for each x_train_single from [1,36,36,28,2] to [[36,36,28], [36,36,28]]
        x_train_single = np.squeeze(x_train[num_volume,:,:,:,:])
        x_train_single = np.split(x_train_single, x_train_single.shape[3], axis=3)
        my_list = [np.squeeze(x_train_single[i]) for i in range(np.shape(x_train_single)[0])]
        
        # y_train_single from [1,36,36,28,2] to [36,36,28]
        y_train_single = np.squeeze(y_train[num_volume,:,:,:,0])

        # call augment
        x_train_single ,y_train_single = augment2(my_list ,y_train_single)

        # x_train_single from [[36,36,28], [36,36,28]] to [1,36,36,28,2]
        x_train_single = np.stack(x_train_single, axis=-1).astype(np.float32)
        x_train_single = np.expand_dims(x_train_single, axis=0)
        # y_train_single from [36,36,28] to [1,36,36,28,2]
        y_train_single = np.stack([y_train_single, y_train_single], axis = -1)
        y_train_single = np.expand_dims(y_train_single, axis=0)
        
        print(y_train.shape, y_train_single.shape)
        # add augmented to x_train and y_train
        x_train = np.concatenate((x_train, x_train_single), axis=0)
        y_train = np.concatenate((y_train, y_train_single), axis=0)


print("train images: ", x_train.shape)
print("train labels: ", y_train.shape)
print("test images: ", x_test.shape)
print("test labels: ", y_test.shape)


### k-fold cross validation
original_shape = x_train.shape
x_train = x_train.reshape((x_train.shape[0], -1))# flatten array to 2d to support scikit split() 
y_train = y_train.reshape((y_train.shape[0], -1))
folds = list(KFold(n_splits=10, shuffle=True, random_state=1).split(x_train, y_train))
x_train = x_train.reshape(original_shape) # reshape the array back to original shape
y_train = y_train.reshape(original_shape)


# model
def get_model():
    size_x, size_y, size_z, num_channels = x_train.shape[1:]

    print("network input size: ", size_x, size_y, size_z, num_channels)

    unet_input = Input(shape=(size_x, size_y, size_z, num_channels))

    conv_1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(unet_input)
    conv_1 = Dropout(0.2)(conv_1)
    conv_1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv_1)
    pool_1 = MaxPooling3D((2, 2, 2), padding = "same")(conv_1)

    #
    conv_2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool_1)
    conv_2 = Dropout(0.2)(conv_2)
    conv_2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv_2)#17 17 12
    pool_2 = MaxPooling3D((2, 2, 2))(conv_2)

    #
    conv_3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool_2)
    conv_3 = Dropout(0.2)(conv_3)
    conv_3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv_3)

    #
    up_1 = UpSampling3D(size=(2, 2, 2))(conv_3)#16 16 12 
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
    conv_7 = Conv3D(num_channels, (1, 1, 1), activation='sigmoid', padding='same', kernel_initializer='he_normal')(conv_6)

    model = Model(inputs=unet_input, outputs=conv_7)
    model.compile(optimizer=Adam(lr=1e-5),loss=dice_loss,metrics=[sensitivity, specificity, precision])
    
    return model


get_model().summary()
model = get_model()  
'''model = load_model('simple_3d_unet_small_volume_3modalities_aug.hdf5',custom_objects={'dice_loss': dice_loss
                                                              ,'sensitivity': sensitivity
                                                              ,'specificity': specificity
                                                              ,'precision': precision})
'''
# training settings

batch_size=1


          
model_checkpoint = ModelCheckpoint('/temp/simple_3d_unet_small_volume_3modalities_aug.hdf5',
                                   verbose=1, save_best_only=True,period = 100)# period is too big for the best checkpoint to be saved


    
model.fit(x_train,y_train,batch_size=1,epochs=15000,verbose=1,callbacks=[model_checkpoint])
#model.fit(x_train,y_train,batch_size=1,epochs=5000,verbose=1)

'''
### callbacks

def get_callbacks(name_weights, patience_lr):
    mcp_save = ModelCheckpoint(name_weights, save_best_only=True
                               , monitor='val_loss', mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='loss', factor=0.1
                                       , patience=patience_lr, verbose=1
                                       , epsilon=1e-4, mode='min')
    return [mcp_save, reduce_lr_loss]


### train model on each fold


gen = ImageDataGenerator()



for j, (train_idx, val_idx) in enumerate(folds):
    
    print('\nFold ',j)
    x_train_cv = x_train[train_idx]
    y_train_cv = y_train[train_idx]
    x_valid_cv = x_train[val_idx]
    y_valid_cv= y_train[val_idx]
    model = get_model()
    name_weights = "simple_3d_unet_small_volume_3modalities_fold" + str(j) + "_weights.h5"
    callbacks = get_callbacks(name_weights = name_weights, patience_lr=10)
    #generator = gen.flow(x_train_cv.reshape(x_train_cv.shape[0], x_train_cv.shape[1] * x_train_cv.shape[2]
    #                                        , x_train_cv.shape[3], x_train_cv.shape[4])
    #                     , y_train_cv.reshape(y_train_cv.shape[0], y_train_cv.shape[1] * y_train_cv.shape[2]
    #                                        , y_train_cv.shape[3], y_train_cv.shape[4])
    #                     , batch_size = batch_size)
    generator = gen.flow(x_train_cv, y_train_cv, batch_size = batch_size)
    
    model.fit_generator(
                generator,
                steps_per_epoch=len(x_train_cv)/batch_size,
                epochs=2000,
                shuffle=True,
                verbose=1,
                validation_data = (x_valid_cv, y_valid_cv),
                callbacks = callbacks)
'''

model.save('simple_3d_unet_small_volume_3modalities_aug.hdf5')
# predict
#prediction = model.predict(x_test, verbose=1)
#save_results(prediction,"data\\test\\prediction\\")



