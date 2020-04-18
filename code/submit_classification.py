from grid import*

if __name__ == '__main__':
    path = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/colorize_images/code/classification_approach.py'
    path_to_train = '/net/projects/data/ImageNet/ILSVRC2012/train' # where to find the training images
    path_to_val = '/net/projects/data/ImageNet/ILSVRC2012/val' # where to find the validation images
    path_to_test = '/net/projects/data/ImageNet/ILSVRC2012/test' # where to find the test images
    batch_size = 10 # chose different batch sizes
    name = 'final' # the name under which all the files/images should be saved
    mode = 'predict' # choose between train and predict
    lr = 0.001
    #learning_rates = [0.01, 0.01, 0.05, 0.08, 0.001] # use these for experimenting during training
    #names = ['0.01', '0.01', '0.05','0.08', '0.001'] # use these for experimenting during training

    env = 'source /net/projects/scratch/winter/valid_until_31_July_2020/asparagus/sharedConda/bin/activate MultiLabel'

    # use the for loop for experimenting during training
    # for lr, name in zip(learning_rates, names):
    #     args = [path_to_train, path_to_val, path_to_test, batch_size, name, mode, lr]

    #     submit_script(path, args, env)

    args = [path_to_train, path_to_val, path_to_test, batch_size, name, mode, lr]
    submit_script(path, args, env)