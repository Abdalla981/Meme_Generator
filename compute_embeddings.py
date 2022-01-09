from pickle import load
from preprocessing.EfficientNet import EfficientNet
from preprocessing.Glove import Glove

'''
This script runs the EfficientNet class and the Glove class for pre-calculation of embeddings
'''

if __name__ == '__main__':
    images_folder = 'dataset/memes3'
    images_output_folder = 'embedding/efficientNet'
    glove_path = 'embedding/glove/glove.42B.300d.txt'
    captions_folder = 'dataset/CaptionsClean3.txt'
    glove_output_file = 'embedding/glove/captionsGlove.pkl'
    not_found_word_file = 'embedding/glove/notFoundWords.txt'
    
    if input('Run EfficientNet?(y/n) ') == 'y':
        image_embedding_obj = EfficientNet(images_folder, models_list=('B0', 'B2'))
        image_embedding_obj.extract_features()
        image_embedding_obj.save_features_to_file(images_output_folder)
        
    if input('Run Glove?(y/n) ') == 'y':
        glove_obj = Glove(captions_folder, glove_path)
        glove_obj.save_vocab_embedding_to_file(glove_output_file, not_found_word_file)
    