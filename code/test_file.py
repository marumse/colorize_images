import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def list_files(dir):
    r = []
    for subdir, dirs, files in os.walk(dir):
        print(subdir)
        print(files[:10])
        if len(r)==10:
            break
        for file in files[:10]:
            filepath = subdir + '/' + file
            r.append(filepath)


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
    print("hi")
    i = 0
    file_list = list_files(directory)
    while True:
        image_batch = []
        label_batch = []
        for b in range(batch_size):
            if i == len(file_list):
                i = 0
                np.random.shuffle(file_list)
            sample = file_list[i]
            i += 1
            print(i)
            #image = cv2.resize(cv2.imread(sample[0]), (224,224))
            image = cv2.resize(cv2.imread(sample), (224,224))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            L = image[:,:,0]
            L = L[:,:,np.newaxis]
            plt.imshow(L)
            plt.savefig('/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/colorize_images/code/L.png')
            ab = image[:,:,1:]
            print("check ab")
            image_batch.append(L)
            print("check L append")
            label_batch.append(ab)
            print("check ab appand")
        yield (np.array(image_batch), np.array(label_batch))


if __name__ == "__main__":
    #print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    
    path = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/colorize_images/code/final_project.py'
    path_to_train = '/net/projects/data/ImageNet/ILSVRC2012/train'
    path_to_val = '/net/projects/data/ImageNet/ILSVRC2012/val'
    batch_size = 2
    file_list = list_files(path_to_train)
    print(file_list)
    # fit model
    training_data = generate_data(path_to_train, batch_size)
    print(training_data)
    print("check")