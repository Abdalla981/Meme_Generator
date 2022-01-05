from utils.Dataset import Dataset
import os


if __name__ == '__main__':
    captions_path = 'old_dataset/Captions2.txt'
    images_path = 'old_dataset/memes2'
    dataset_obj = Dataset(captions_path, images_path, clean_dataset=True)
    print(len(dataset_obj.captions), len(dataset_obj.images))