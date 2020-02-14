import numpy as np 
#import tensorflow as tf 
import keras

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Flatten, Activation, BatchNormalization
#from keras.layers import Dense


def generate_dataset(path_to_train, path_to_val, path_to_test, path_to_save, batch_size=3):
    
    datagen = ImageDataGenerator() #featurewise_center=True, featurewise_std_normalization=True)
    # load and iterate training dataset
    train_it = datagen.flow_from_directory(path_to_train, target_size=(224,224), save_to_dir = path_to_save, class_mode=None, batch_size=batch_size) # for class_mode=None we need subfolders in dir?
    # load and iterate validation dataset
    val_it = datagen.flow_from_directory(path_to_val, target_size=(224,224), class_mode=None, batch_size=batch_size)
    # load and iterate test dataset
    test_it = datagen.flow_from_directory(path_to_test, target_size=(224,224), class_mode=None, batch_size=batch_size)
    
    return train_it, val_it, test_it


if __name__ == "__main__":

    path_to_save = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/LabelFiles/colorize_images/save/'
    path_to_test = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/LabelFiles/colorize_images/test/'
    path_to_train = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/LabelFiles/colorize_images/train/'
    path_to_val = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/LabelFiles/colorize_images/validation/'

    train_it, val_it, test_it = generate_dataset(path_to_train, path_to_val, path_to_test, path_to_save, batch_size=3)
    # confirm the iterator works
    batchX = train_it.next()
    print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))
    
    # define model
    model = Sequential()
    model.add(Conv2D(3, (3,3), padding='same', input_shape=batchX.shape[1:]))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    
    # compile model
    model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])
    # fit model
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    model.fit_generator(train_it, steps_per_epoch=16, validation_data=val_it, validation_steps=8)
    # save weights
    #model.save_weights('first_try.h5')
    # evaluate model
    loss = model.evaluate_generator(test_it, steps=24)
    # make a prediction
    #yhat = model.predict_generator(predict_it, steps=24)