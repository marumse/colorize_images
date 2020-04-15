import numpy as np
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt
import keras
from keras import backend as K

from keras.models import Sequential

from keras.activations import softmax

from keras.layers import Conv2D, Conv2DTranspose, Activation, BatchNormalization, Softmax, Multiply

from grid import*
from submit_model import*

def list_files(dir):
    """ List all files in a given directory including all subdirectories.
    Args:       path to a directory
    Return:     list with all complete file paths
    """
    r = []
    for subdir, dirs, files in os.walk(dir):
        for file in files[:5]:
            filepath = subdir + '/' + file
            r.append(filepath)
            if len(r)==5000: # set lower for prediction 
                break
    return r

def cantor_pairing_dict():
    """ Create dictionary that pairs the values from the cantor pairing with indices for the one hot encoding.
    Args:       None
    Return:     the dict for possible cantor values (121 different combinations possible)
    """
    possible_a = list(range(0,255,25))
    possible_b = list(range(0,255,25))
    i = 0
    dicts = {}
    for a in possible_a:
        for b in possible_b:
            pairing = ((a + b) * (a + b + 1)) / 2 + b
            dicts[pairing] = i
            i += 1
    return dicts

def one_hot_encoding(cantor):
    """ Get one-hot-encoding for ab target image.
    Args:       ab layers from input images
    Return:     one-hot-encoding with shape width x height x number of colors (224 x 224 x 121)
    """
    one_hot = np.zeros((cantor.shape[0], cantor.shape[1], 121))
    for i, unique_value in enumerate(np.unique(cantor)):
        index = dicts[unique_value]
        one_hot[:, :, index][cantor == unique_value] = 1
    return one_hot

def cantor_pairing(ab):
    a = ab[:,:,0].astype(np.uint32)
    b = ab[:,:,1].astype(np.uint32)
    c = ((a + b) * (a + b + 1)) / 2 + b
    return c

def reverse_cantor(z):
    t = np.floor((-1 + np.sqrt(1 + 8 * z))/2)
    x = t * (t + 3) / 2 - z
    y = z - t * (t + 1) / 2
    x = x[:,:,np.newaxis]
    y = y[:,:,np.newaxis]
    return np.concatenate((x, y), axis=2)

def generate_data(batch_size, file_list):
    """ Replaces Keras' native ImageDataGenerator.
        This function is a data generator that loads a costumized version of some data. More precisely, it loads images,
        tranforms them into LAB color space and returns the first layer as the input and the other two layers as the target for the model.
        Args:       batch_size 
                    file_list containing all image paths
        Return:     a tuple containing a numpy array with the inputs and a second numpy array with the targets
    """
    i = 0
    while True:
        image_batch = []
        label_batch = []

        for b in range(batch_size):
            # shuffle data when all files have been seen
            if i == len(file_list):
                i = 0
                np.random.shuffle(file_list)
            sample = file_list[i]
            i += 1
            image = cv2.resize(cv2.imread(sample), (224,224))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            # split the image into the L layer for the input and ab layers for the target 
            L = image[:,:,0][:,:,np.newaxis]
            ab = image[:,:,1:]
            # reduce the number of colors to 121 (11 different a and b values respectively)
            ab = (ab//25)*25
            # use cantor pairing to combine layer a and b
            cantor = cantor_pairing(ab)
            # make a one-hot-encoding out of the cantor labels
            hot = one_hot_encoding(cantor)
            # append both to the corresponding lists
            image_batch.append(L)
            label_batch.append(hot)
        yield (np.array(image_batch), np.array(label_batch))

def generate_test_data(test_batch, file_list):
    """ Data generator for the test dataset.
        This functions works very similar to the generate_data funtion only adjusted for the test data, which is only used for prediction.
        Args:       test_batch how many predictions do we want to make
                    file_list containing all image paths
        Return:     a tuple containing a numpy array with the inputs and a second numpy array with the targets
    """
    i = 0
    image_batch = []
    label_batch = []
    # shuffle data to get different test images each time
    np.random.shuffle(file_list)
    for b in range(test_batch):
        sample = file_list[i]
        i += 1
        # read in the image and convert it to LAB color space
        image = cv2.resize(cv2.imread(sample), (224,224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        # the first layer of the image will be the input
        L = image[:,:,0][:,:,np.newaxis]
        ab = image[:,:,1:]
        # reduce the number of colors to 121 (11 different a and b values respectively)
        ab = (ab//25)*25
        # use cantor pairing to combine layer a and b
        cantor = cantor_pairing(ab)
        # make a one-hot-encoding out of the cantor labels
        hot = one_hot_encoding(cantor)
        # append both to the corresponding lists
        image_batch.append(L)
        label_batch.append(hot)
    return np.array(image_batch), np.array(label_batch)

def softMaxAxis2(x):
    return softmax(x, axis=2)

def create_model():
    """ Built the model and compile it.
        Args:       None
        Return:     the compiled model
    """
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
    model.add(Conv2D(121, (1,1), strides = 1, dilation_rate = 1))
    model.add(Softmax())

    # decoding layer
    model.add(Conv2DTranspose(121, (3,3), strides = 16, padding = 'same'))
    model.add(Conv2D(121, (1,1), strides = 1, dilation_rate = 1, activation = softMaxAxis2))
    
    # compile model
    sgd = keras.optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True, clipnorm=5.)
    model.compile(optimizer=sgd, loss=keras.losses.categorical_crossentropy)

    model.summary()

    return model

def make_prediction(test_batch, test_files, name):
    """ Make a predition for an unseen batch of test images.
        Predictions are made with the model after it was trained and then saved for visual inspection.
        Args:       test_batch how many predictions do we want to make
                    file_list containing all image paths
        Return:     None
    """
    # make predictions of a small test sample
    test_in, test_out = generate_test_data(test_batch, test_files)
    prediction = model.predict_on_batch(test_in)
    for i in range(test_batch):
        original = np.concatenate((test_in[i], test_out[i]), axis=2)
        # save the image in BGR color space in order to display it straight away
        original_BGR = cv2.cvtColor(original, cv2.COLOR_LAB2BGR)
        cv2.imwrite('/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/colorize_images/results/predictions/orig_'+ str(i) + name +'.png', original_BGR)
        predicted = np.concatenate((test_in[i], prediction[i]), axis=2)
        # same for the predicted image
        # for some reason this yields a black BGR image - save the LAB image, load it again and then transform it to BGR works fine
        predicted_BGR = cv2.cvtColor(predicted, cv2.COLOR_LAB2BGR)
        cv2.imwrite('/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/colorize_images/results/predictions/pred_' + str(i) + name +'.png', predicted)
        #cv2.imwrite('/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/colorize_images/results/predictions/pred_1_BGR.png', predicted_BGR)

def plot_history(history, name):
    
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
    plt.savefig('/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/colorize_images/results/'+ name +'.png')

def save_history(history, name):
    #convert the history.history dict to a pandas DataFrame   
    hist_df = pd.DataFrame(history.history) 

    # and save to csv
    hist_csv_file = name +'.csv'
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)


if __name__ == "__main__":
    
    # collect arguments from submit_script
    args = typecast(sys.argv[1:])
    path_to_train = args[0]
    path_to_val = args[1]
    path_to_test = args[2]
    batch_size = args[3]
    name = args[4]
    mode = args[5]
    test_batch = 5
    global dicts
    # create dictionary for cantor values and one-hot-indices
    dicts = cantor_pairing_dict()

    if mode == 'train':
        # collect all file paths to the train, validation and test images
        train_files = list_files(path_to_train)
        val_files = list_files(path_to_val)

        # create the model
        model = create_model()
        
        # generate the data with the costumized generator
        train_gen = generate_data(batch_size, train_files)
        val_gen = generate_data(batch_size, val_files)

        # fit model
        history = model.fit_generator(train_gen, steps_per_epoch=250, epochs=4, validation_data=val_gen, validation_steps=1)

        # save weights
        model.save_weights('/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/colorize_images/code/'+ name +'.h5')

        # plot and save the accuracy and loss values
        plot_history(history, name)
        # save history too
        save_history(history, name)

    if mode == 'predict':
        # load the test files
        test_files = list_files(path_to_test)
        # create and compile the model
        model = create_model()
        # load the weights with which the predictions should be done
        model.load_weights('/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/colorize_images/code/'+ name +'.h5')        
        # make a prediction and save the image
        make_prediction(test_batch, test_files, name)