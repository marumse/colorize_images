import numpy as np
#import tensorflow as tf
#from skimage import color
import cv2
import os
import matplotlib.pyplot as plt
import keras
from keras import backend as K 
#from tensorflow.keras.preprocessing.image import ImageDataGenerator

#from tensorflow.keras.models import Sequential

#from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Activation, BatchNormalization, Softmax, Multiply

from keras.models import Sequential

from keras.layers import Conv2D, Conv2DTranspose, Activation, BatchNormalization, Softmax, Multiply

#from tensorflow.keras.optimizers import Adam

from grid import*
from submit_model import*

def list_files(dir):
    r = []
    for subdir, dirs, files in os.walk(dir):
        if len(r)==10:
            break
        for file in files[:1]:
            filepath = subdir + '/' + file
            r.append(filepath)

    print(len(r))
    return r

def generate_val_data(batch_size, file_list):
    i = 0
    image_batch = []
    label_batch = []
    for b in range(batch_size):
        if i == len(file_list):
            i = 0
            np.random.shuffle(file_list)
        sample = file_list[i]
        i += 1
        print(sample)
        #image = cv2.resize(cv2.imread(sample[0]), (224,224))
        image = cv2.resize(cv2.imread(sample), (224,224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        L = image[:,:,0]
        L = L[:,:,np.newaxis]
        #plt.imshow(L)
        #plt.savefig('/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/colorize_images/code/L.png')
        ab = image[:,:,1:]
        image_batch.append(L)
        label_batch.append(ab)
    return (np.array(image_batch), np.array(label_batch))



def generate_data(batch_size, file_list):
    """ Replaces Keras' native ImageDataGenerator.
        code snippet from: https://stackoverflow.com/questions/46493419/use-a-generator-for-keras-model-fit-generator
    """
    i = 0
    j = 0
    while True:
        image_batch = []
        label_batch = []

        for b in range(batch_size):
            if i == len(file_list):
                i = 0
                np.random.shuffle(file_list)
            sample = file_list[i]
            i += 1
            print(sample)
            #image = cv2.resize(cv2.imread(sample[0]), (224,224))
            image = cv2.resize(cv2.imread(sample), (224,224))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            # split the image into the L layer for the input and ab layers for the target 
            L = image[:,:,0]
            L = L[:,:,np.newaxis]
            ab = image[:,:,1:]
            image_batch.append(L)
            label_batch.append(ab)
        yield (np.array(image_batch), np.array(label_batch))

def create_model():
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
    model.add(Conv2D(256, (3,3), strides = 2, dilation_rate = 1, padding='same'))
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
    model.add(Conv2DTranspose(313, (3,3), strides = 16, padding = 'same'))
    model.add(Conv2D(2, (1,1), strides = 1, dilation_rate = 1))
    
    # compile model
    sgd = keras.optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True, clipnorm=5.)

    model.compile(optimizer=sgd, loss=keras.losses.mean_squared_error)

    return model


if __name__ == "__main__":
    # collect arguments from submit_script
    args = typecast(sys.argv[1:])
    path_to_train = args[0]
    path_to_val = args[1]
    batch_size = args[2]

    # get all the file paths to the train and validation images
    train_files = list_files(path_to_train)
    val_files = list_files(path_to_val)
    
    # create the model
    model = create_model()
    
    # generate the data with the costumized generator
    train_gen = generate_data(batch_size, train_files)
    val_gen = generate_data(batch_size, val_files)
    test_gen = generate_data(batch_size, val_files[:10])

    # fit model
    history = model.fit_generator(train_gen, steps_per_epoch=40, epochs=1, validation_data=val_gen, validation_steps=1)
    print(history.history)

    prediction = model.predict_generator(test_gen, steps=1, max_queue_size=10)
    print(prediction.shape)
    # save weights
    model.save_weights('/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/colorize_images/code/second_try.h5')
    
    # evaluate model
    plt.figure(facecolor='white')

    plt.plot(history.history['loss'], label="loss", color="blue")
    plt.plot(history.history['val_loss'], label="val_loss", color="red")

    plt.title('Loss History')
    plt.ylabel('loss')
    plt.xlabel('epoch')

    plt.legend(['train', 'valid'], loc='lower left')

    plt.ylim(0)
    plt.xticks(np.arange(0, 3 + 1, 5))
    plt.grid()
    plt.show()
    plt.savefig('/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/colorize_images/code/fig_model2.png')
