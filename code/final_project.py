import numpy as np
import tensorflow as tf
#from skimage import color
import cv2
import os
import matplotlib.pyplot as plt

#from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Activation, BatchNormalization, Softmax, Multiply

#from tensorflow.keras.optimizers import Adam

from grid import*
from submit_model import*

def list_files(dir):
    print("check3.1.1")
    r = []
    for subdir, dirs, files in os.walk(dir):
        print(subdir)
        print(files[:10])
        for file in files[:10]:
            filepath = subdir + '/' + file
            r.append(filepath)
            print(len(r))

    # for root, dirs, files in os.walk(dir):
    #     print("check3.1.2")
    #     print(root)
    #     print(dirs)
    #     print(files[:10])
    #     for name in files[:10]:
    #         print("check3.1.3")
    #         r.append(os.path.join(dirs, name))
            
    print(len(r))
    return r

def generate_data(directory, batch_size):
    """ Replaces Keras' native ImageDataGenerator.
        code snippet from: https://stackoverflow.com/questions/46493419/use-a-generator-for-keras-model-fit-generator
    """
    print("check3.1")
    i = 0
    file_list = list_files(directory)
    print("check3.2")
    while True:
        print("check3.3")
        image_batch = []
        label_batch = []
        for b in range(batch_size):
            if i == len(file_list):
                i = 0
                np.random.shuffle(file_list)
            sample = file_list[i]
            i += 1
            #image = cv2.resize(cv2.imread(sample[0]), (224,224))
            image = cv2.resize(cv2.imread(directory + sample), (224,224))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            L = image[:,:,0]
            L = L[:,:,np.newaxis]
            ab = image[:,:,1:]
            image_batch.append(L)
            label_batch.append(ab)
            print("check3.4")
        yield (np.array(image_batch), np.array(label_batch))


if __name__ == "__main__":
    #print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    
    tf.debugging.set_log_device_placement(True)
    print("checkpoint1")
    args = typecast(sys.argv[1:])
    path_to_train = args[0]
    path_to_val = args[1]

    batch_size = args[2]

    print(tf.test.is_gpu_available())
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
    #model.add(Multiply())
    model.add(Softmax())

    # decoding layer
    model.add(Conv2DTranspose(313, (4,4), strides = 16, padding = 'same'))
    model.add(Conv2D(2, (1,1), strides = 1, dilation_rate = 1))
    print("checkpoint2")
    # compile model
    sgd = tf.keras.optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True, clipnorm=5.)
    mse = tf.keras.losses.MeanSquaredError()
    model.compile(optimizer=sgd, loss=mse)
    #print(model.summary())
    print("checkpoint3")
    # fit model
    history = model.fit_generator(generate_data(path_to_train, batch_size), steps_per_epoch=400, epochs=5, validation_data=generate_data(path_to_val,batch_size), validation_steps=8)
    print(history.history)
    # save weights
    model.save_weights('/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/colorize_images/code/first_try.h5')
    # evaluate model
    print("checkpoint4")
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
    plt.savefig('/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/colorize_images/code/fig_model1.png')
    #plt.savefig('C:/Users/Acer/colorize_images/code/fig_model1.png')
