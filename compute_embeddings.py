from numpy import mod
from preprocessing.EfficientNet import EfficientNet
from preprocessing.Dataset import Dataset

if __name__ == '__main__':
    images_folder = 'dataset/memes3'
    output_folder = 'embedding/efficientNet'
    image_embedding_obj = EfficientNet(images_folder, output_folder, 
                                       models_list=('B0', 'B2'))
    image_embedding_obj.extract_features()
    image_embedding_obj.save_features_to_file()
    