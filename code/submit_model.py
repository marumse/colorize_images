from grid import*

if __name__ == '__main__':
    path = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/colorize_images/code/final_project.py'
    path_to_train = '/net/projects/data/ImageNet/ILSVRC2012/train'
    path_to_val = '/net/projects/data/ImageNet/ILSVRC2012/val'
    path_to_test = '/net/projects/data/ImageNet/ILSVRC2012/test'
    batch_size = 5
    args = [path_to_train, path_to_val, path_to_test, batch_size]

    env = 'source /net/projects/scratch/winter/valid_until_31_July_2020/asparagus/sharedConda/bin/activate MultiLabel'

    submit_script(path, args, env)
