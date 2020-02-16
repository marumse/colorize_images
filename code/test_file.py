import numpy as np 
import tensorflow as tf 
#from skimage import color
import cv2

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization, Softmax, Multiply

from tensorflow.keras.optimizers import Adam

#from tensorflow.keras.losses import KLD


def transform_image_L(img):
    img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    L = img_lab[:,:,0]
    img_new = L[:,:,np.newaxis]
    return img_new

def transform_image_ab(img):
    img_new = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    return img_new[:,:,1:]


if __name__ == "__main__":

    #from PIL import ImageFile
    #ImageFile.LOAD_TRUNCATED_IMAGES = True

    #path_to_save = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/LabelFiles/colorize_images/save/'
    #path_to_test = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/LabelFiles/colorize_images/test/'
    #path_to_train = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/LabelFiles/colorize_images/train/'
    #path_to_val = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/LabelFiles/colorize_images/validation/'
    path_to_train = 'C:/Users/Acer/colorize_images/ex_pics/train/'
    batch_size = 3

    # data generator for large datasets
    # featurewise_center=True, featurewise_std_normalization=True)
    # if we want this we need .fit()
    datagen_imgs = ImageDataGenerator(preprocessing_function = transform_image_L)
    # separat generator for labels
    datagen_target = ImageDataGenerator(preprocessing_function = transform_image_ab)
    # load and iterate training dataset
    train_it = datagen_imgs.flow_from_directory(path_to_train, target_size=(224,224), class_mode=None, batch_size=batch_size) # for class_mode=None we need subfolders in dir?
    target_it = datagen_target.flow_from_directory(path_to_train, target_size=(224,224), class_mode=None, batch_size=batch_size)
    # load and iterate validation dataset
    #val_it = datagen.flow_from_directory(path_to_val, target_size=(224,224), class_mode=None, batch_size=batch_size)
    # load and iterate test dataset
    #test_it = datagen.flow_from_directory(path_to_test, target_size=(224,224), class_mode=None, batch_size=batch_size)
    # confirm the iterator works
    batchX = train_it.next()
    print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))
    #L, a, b = tf.unstack(batchX, axis = 3)
    #print(L.shape) # (32,224,224)
    #print(a.shape) # (32,224,224)
    #print(b.shape) # (32,224,224)
    #ab = tf.stack([a,b], axis = -1)
    #print(ab.shape) # (32,224,224,2)
    #L = tf.expand_dims(L, axis = 3)
    #print(L.shape) # (32,224,224,1)
    
    # define model
    model = Sequential()
    # conv1
    model.add(Conv2D(64, (3,3), padding='same', input_shape=[224,224,1]))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3,3), strides = 2, padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    # conv2
    model.add(Conv2D(128, (3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3,3), strides = 2, padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    # conv3
    model.add(Conv2D(256, (3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3,3), strides = 2, padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    # conv4
    model.add(Conv2D(512, (3,3), strides = 1, dilation_rate = 1, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3,3), strides = 1, dilation_rate = 1, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3,3), strides = 1, dilation_rate = 1, padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    # conv5
    model.add(Conv2D(512, (3,3), strides = 1, dilation_rate = 2, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3,3), strides = 1, dilation_rate = 2, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3,3), strides = 1, dilation_rate = 2, padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    # conv6
    model.add(Conv2D(512, (3,3), dilation_rate = 2, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3,3), dilation_rate = 2, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3,3), dilation_rate = 2, padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    # conv7
    model.add(Conv2D(512, (3,3), dilation_rate = 1, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3,3), dilation_rate = 1, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3,3), dilation_rate = 1, padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    # conv8
    model.add(Conv2D(256, (4,4), strides = 2, dilation_rate = 1, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3,3), dilation_rate = 1, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3,3), dilation_rate = 1, padding='same'))
    model.add(Activation('relu'))

    # softmax layer
    model.add(Conv2D(313, (1,1), strides = 1, dilation_rate = 1))
    #model.add(Multiply())
    model.add(Softmax())

    # decoding layer
    model.add(Conv2D(2, (1,1), strides = 1, dilation_rate = 1))
    
    # compile model
    model.compile(optimizer=Adam(0.01), loss = 'mean_squared_error')
    # fit model
    model.fit_generator(train_it, target_it)


    # save weights
    #model.save_weights('first_try.h5')
    # evaluate model
    #loss = model.evaluate_generator(test_it, steps=24)
    # make a prediction
    #yhat = model.predict_generator(predict_it, steps=24)

    import numpy as np 
import tensorflow as tf 
#from skimage import color
import cv2
import os
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization, Softmax, Multiply

from tensorflow.keras.optimizers import Adam




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
    return datagen, train_it, val_it, test_it

def generate_data(directory, batch_size):
    """Replaces Keras' native ImageDataGenerator from: https://stackoverflow.com/questions/46493419/use-a-generator-for-keras-model-fit-generator"""
    i = 0
    file_list = os.listdir(directory)
    while True:
        image_batch = []
        label_batch = []
        for b in range(batch_size):
            if i == len(file_list):
                i = 0
                random.shuffle(file_list)
            sample = file_list[i]
            i += 1
            #image = cv2.resize(cv2.imread(sample[0]), (224,224))
            image = cv2.resize(cv2.imread(directory + sample), (224,224))
            L = image[:,:,0]
            L = L[:,:,np.newaxis]
            ab = image[:,:,1:]
            image_batch.append(L)
            label_batch.append(ab)

        yield (np.array(image_batch), np.array(label_batch))


if __name__ == "__main__":

    

    #from PIL import ImageFile
    #ImageFile.LOAD_TRUNCATED_IMAGES = True

    #path_to_save = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/LabelFiles/colorize_images/save/'
    #path_to_test = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/LabelFiles/colorize_images/test/'
    #path_to_train = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/LabelFiles/colorize_images/train/'
    #path_to_val = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/LabelFiles/colorize_images/validation/'
    path_to_train = 'C:/Users/Acer/colorize_images/ex_pics/train/train/'
    path_to_val = 'C:/Users/Acer/colorize_images/ex_pics/validation/validation/'
    batch_size = 3

    # # data generator for large datasets
    # # featurewise_center=True, featurewise_std_normalization=True)
    # # if we want this we need .fit()
    # datagen = ImageDataGenerator(preprocessing_function = transform_image) 
    # # load and iterate training dataset
    # train_it = datagen.flow_from_directory(path_to_train, target_size=(224,224), class_mode=None, batch_size=batch_size) # for class_mode=None we need subfolders in dir?
    # # load and iterate validation dataset
    # val_it = datagen.flow_from_directory(path_to_train, target_size=(224,224), class_mode=None, batch_size=batch_size)
    # # load and iterate test dataset
    # #test_it = datagen.flow_from_directory(path_to_test, target_size=(224,224), class_mode=None, batch_size=batch_size)
    # # confirm the iterator works
    # batchX = train_it.next()
    # print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))
    # L, a, b = tf.unstack(batchX, axis = 3)
    # #print(L.shape) # (32,224,224)
    # #print(a.shape) # (32,224,224)
    # #print(b.shape) # (32,224,224)
    # ab = tf.stack([a,b], axis = -1)
    # #print(ab.shape) # (32,224,224,2)
    # L = tf.expand_dims(L, axis = 3)
    # #print(L.shape) # (32,224,224,1)
    
    # define model
    model = Sequential()
    # conv1
    model.add(Conv2D(64, (3,3), padding='same', input_shape=(224,224,1)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3,3), strides = 2, padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    # conv2
    model.add(Conv2D(128, (3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3,3), strides = 2, padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    # conv3
    model.add(Conv2D(256, (3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3,3), strides = 2, padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    # conv4
    model.add(Conv2D(512, (3,3), strides = 1, dilation_rate = 1, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3,3), strides = 1, dilation_rate = 1, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3,3), strides = 1, dilation_rate = 1, padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    # conv5
    model.add(Conv2D(512, (3,3), strides = 1, dilation_rate = 2, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3,3), strides = 1, dilation_rate = 2, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3,3), strides = 1, dilation_rate = 2, padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    # conv6
    model.add(Conv2D(512, (3,3), dilation_rate = 2, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3,3), dilation_rate = 2, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3,3), dilation_rate = 2, padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    # conv7
    model.add(Conv2D(512, (3,3), dilation_rate = 1, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3,3), dilation_rate = 1, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3,3), dilation_rate = 1, padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    # conv8
    model.add(Conv2D(256, (4,4), strides = 2, dilation_rate = 1, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3,3), dilation_rate = 1, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3,3), dilation_rate = 1, padding='same'))
    model.add(Activation('relu'))

    # softmax layer
    model.add(Conv2D(313, (1,1), strides = 1, dilation_rate = 1))
    model.add(Multiply())
    model.add(Softmax())

    # decoding layer
    model.add(Conv2D(2, (1,1), strides = 1, dilation_rate = 1))
    
    # compile model
    sgd = tf.keras.optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True, clipnorm=5.)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    print(model.summary())
    # fit model
    model.fit_generator(generate_data(path_to_train, batch_size), steps_per_epoch=3, validation_data=generate_data(path_to_val,batch_size), validation_steps=8)


    # save weights
    #model.save_weights('first_try.h5')
    # evaluate model
    #loss = model.evaluate_generator(test_it, steps=24)
    # make a prediction
    #yhat = model.predict_generator(predict_it, steps=24)