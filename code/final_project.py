import numpy as np 
#import tensorflow as tf 
#from skimage import color
import keras

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Flatten, Activation, BatchNormalization
#from keras.layers import Dense


def transform_image(img):
    img_new = img.copy()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img_new[i,j,:] = rgb_to_lab(img[i,j,:])
    return img_new

def rgb_to_lab(inputColor):

   num = 0
   RGB = [0, 0, 0]

   for value in inputColor :
       value = float(value) / 255

       if value > 0.04045 :
           value = ( ( value + 0.055 ) / 1.055 ) ** 2.4
       else :
           value = value / 12.92

       RGB[num] = value * 100
       num = num + 1

   XYZ = [0, 0, 0,]

   X = RGB [0] * 0.4124 + RGB [1] * 0.3576 + RGB [2] * 0.1805
   Y = RGB [0] * 0.2126 + RGB [1] * 0.7152 + RGB [2] * 0.0722
   Z = RGB [0] * 0.0193 + RGB [1] * 0.1192 + RGB [2] * 0.9505
   XYZ[ 0 ] = round( X, 4 )
   XYZ[ 1 ] = round( Y, 4 )
   XYZ[ 2 ] = round( Z, 4 )

   XYZ[ 0 ] = float( XYZ[ 0 ] ) / 95.047         # ref_X =  95.047   Observer= 2°, Illuminant= D65
   XYZ[ 1 ] = float( XYZ[ 1 ] ) / 100.0          # ref_Y = 100.000
   XYZ[ 2 ] = float( XYZ[ 2 ] ) / 108.883        # ref_Z = 108.883

   num = 0
   for value in XYZ :

       if value > 0.008856 :
           value = value ** ( 0.3333333333333333 )
       else :
           value = ( 7.787 * value ) + ( 16 / 116 )

       XYZ[num] = value
       num = num + 1

   Lab = [0, 0, 0]

   L = ( 116 * XYZ[ 1 ] ) - 16
   a = 500 * ( XYZ[ 0 ] - XYZ[ 1 ] )
   b = 200 * ( XYZ[ 1 ] - XYZ[ 2 ] )

   Lab [ 0 ] = round( L, 4 )
   Lab [ 1 ] = round( a, 4 )
   Lab [ 2 ] = round( b, 4 )

   return Lab

def generate_dataset(path_to_train, path_to_val, path_to_test, path_to_save, batch_size=3):
    
    datagen = ImageDataGenerator(preprocessing_function = transform_image) #featurewise_center=True, featurewise_std_normalization=True)
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

    train_it, val_it, test_it = generate_dataset(path_to_train, path_to_val, path_to_test, path_to_save, batch_size=3)
    # confirm the iterator works
    batchX = train_it.next()
    print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))
    
    # define model
    model = Sequential()
    model.add(Conv2D(32, (3,3), padding='same', input_shape=batchX.shape[1:]))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    
    # compile model
    model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])
    # fit model

    model.fit_generator(train_it, steps_per_epoch=16, validation_data=val_it, validation_steps=8)
    # save weights
    #model.save_weights('first_try.h5')
    # evaluate model
    loss = model.evaluate_generator(test_it, steps=24)
    # make a prediction
    #yhat = model.predict_generator(predict_it, steps=24)