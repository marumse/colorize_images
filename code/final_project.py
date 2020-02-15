import numpy as np 
import tensorflow as tf 
#from skimage import color
import cv2

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Activation, BatchNormalization
#from keras.layers import Dense


def transform_image(img):
    img_new = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    return img_new


def generate_dataset(path_to_train, path_to_val, path_to_test, path_to_save, batch_size=3):
    # featurewise_center=True, featurewise_std_normalization=True)
    # if we want this we need .fit()
    datagen = ImageDataGenerator(preprocessing_function = transform_image) 
    # load and iterate training dataset
    train_it = datagen.flow_from_directory(path_to_train, target_size=(224,224), save_to_dir = path_to_save, class_mode=None, batch_size=batch_size) # for class_mode=None we need subfolders in dir?
    # load and iterate validation dataset
    val_it = datagen.flow_from_directory(path_to_val, target_size=(224,224), class_mode=None, batch_size=batch_size)
    # load and iterate test dataset
    test_it = datagen.flow_from_directory(path_to_test, target_size=(224,224), class_mode=None, batch_size=batch_size)
    
    # change color space to lab
    return train_it, val_it, test_it


if __name__ == "__main__":

    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    path_to_save = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/LabelFiles/colorize_images/save/'
    path_to_test = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/LabelFiles/colorize_images/test/'
    path_to_train = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/LabelFiles/colorize_images/train/'
    path_to_val = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/LabelFiles/colorize_images/validation/'

    train_it, val_it, test_it = generate_dataset(path_to_train, path_to_val, path_to_test, path_to_save, batch_size=32)
    # confirm the iterator works
    batchX = train_it.next()
    print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))
    L, a, b = tf.unstack(batchX, axis = 3)
    print('input_L: ' + L.shape)
    print('output_ab: ' + a.shape)
    print('output_ab: ' + b.shape)

    
    # # define model
    # model = Sequential()
    # model.add(Conv2D(32, (3,3), padding='same', input_shape=batchX.shape[1:]))
    # model.add(Activation('relu'))
    # model.add(BatchNormalization())
    # model.add(MaxPooling2D(pool_size=(2,2)))
    # model.add(Flatten())
    
    # # compile model
    # model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])
    # # fit model

    # model.fit_generator(train_it, steps_per_epoch=16, validation_data=val_it, validation_steps=8)
    # # save weights
    # #model.save_weights('first_try.h5')
    # # evaluate model
    # loss = model.evaluate_generator(test_it, steps=24)
    # # make a prediction
    # #yhat = model.predict_generator(predict_it, steps=24)