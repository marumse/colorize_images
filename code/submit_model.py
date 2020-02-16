from grid import*

if __name__ == '__main__':
    path = '/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/asparagus/colorize_images/code/final_project.py'
    path_to_train = '/net/projects/data/ImageNet/ILSVRC2012/train'
    path_to_val = '/net/projects/data/ImageNet/ILSVRC2012/val'
    batch_size = 25
    args = [path_to_train, path_to_val, batch_size]
   # env = 'source stack/bin/activate'

    submit_script(path, args)
