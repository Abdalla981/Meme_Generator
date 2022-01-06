import enum
import numpy as np
from math import floor
from utils.Dataset import Dataset
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

'''
This class inherits the Dataset class and prepares the dataset to be passed throught the model.

Methods:
- process_dataset: prepares the dataset for training
- captions_to_list: returns a list of all captions
- add_tokens_to_caption: Adds start and end token to each caption
- captions_tokenizer: fits a tokenizer on the training data and returns the tokenizer
- tallest_seq_length: returns the length of the tallest sequence
- split_dataset: splits the dataset into train, validation and test sets
- create_sequences: creates an array of sequences (image, sequence[:i], next_word[i])
- load_photo_features: TODO
- print_dataset_info: prints useful info on the loaded dataset
'''

class DatasetProcessor(Dataset):
    def __init__(self, captions_path, images_path, start_token='startseq', end_token='endseq',
                 split_value=(0.8, 0.1)):
        super().__init__(captions_path, images_path)
        self.start_token = start_token
        self.end_token = end_token
        self.split_value = split_value
        self.num_of_train_images = floor(self.num_of_samples * self.split_value[0])
        self.num_of_validation_images = floor(self.num_of_samples * self.split_value[1])
        self.tokenizer, self.num_of_vocab = None, None
        self.train_captions_list, self.max_seq_length = None, None
        self.train_captions, self.train_images = None, None
        self.validation_captions, self.validation_images = None, None
        self.test_captions, self.test_images = None, None
        
    def process_dataset(self):
        self.add_tokens_to_caption()
        (train, validation, test) = self.split_dataset()
        self.train_captions, self.train_images = train
        self.validation_captions, self.validation_images = validation
        self.test_captions, self.test_images = test
        self.train_captions_list = self.captions_to_list(captions_dict=self.train_captions)
        self.tokenizer = self.captions_tokenizer()
        self.num_of_vocab = len(self.tokenizer.word_index) + 1
        self.max_seq_length = self.tallest_seq_length()
        
    def captions_to_list(self, captions_dict=None):
        captions_list = []
        if captions_dict is None:
            captions_dict = self.captions
        for captions in captions_dict.values():
            [captions_list.append(caption) for caption in captions]
        return captions_list
    
    def add_tokens_to_caption(self):
        for name, captions in self.captions.items():
            for i, caption in enumerate(captions):
                tokened = ' '.join([self.start_token] + caption.split() + [self.end_token])
                self.captions[name][i] = tokened
    
    def captions_tokenizer(self):
        tokenizer = Tokenizer(oov_token=1)
        tokenizer.fit_on_texts(self.train_captions_list)
        return tokenizer
    
    def split_dataset(self):
        train_captions, train_images, test_captions, test_images = {}, {}, {}, {}
        validation_captions, validation_images = {}, {}
        names = list(self.captions.keys())
        np.random.shuffle(names)
        train_index = self.num_of_train_images
        validation_index = self.num_of_train_images + self.num_of_validation_images
        for name in names[:train_index]:
            train_captions[name] = self.captions[name]
            train_images[name] = self.images[name]
        for name in names[train_index:validation_index]:
            validation_captions[name] = self.captions[name]
            validation_images[name] = self.images[name]
        for name in names[validation_index:]:
            test_captions[name] = self.captions[name]
            test_images[name] = self.images[name]
        return (train_captions, train_images), (validation_captions, validation_images), (test_captions, test_images)

    def create_sequences(self, captions_dict, images_dict):
        X1, X2, y = [], [], []
        for name, captions in captions_dict.items():
            for caption in captions:
                sequence = self.tokenizer.texts_to_sequences([caption])[0]
                for i in range(1, len(sequence)):
                    in_sequence, out_sequence = sequence[:i], sequence[i]
                    in_sequence = pad_sequences([in_sequence], maxlen=self.max_seq_length)[0]
                    out_sequence = to_categorical([out_sequence], num_classes=self.num_of_vocab)
                    X1.append(images_dict[name])
                    X2.append(in_sequence)
                    y.append(out_sequence)
        return np.array(X1), np.array(X2), np.array(y)
    
    def tallest_seq_length(self):
        return max(len(caption.split()) for caption in self.train_captions_list)
    
    def load_photo_features(self):
        pass

    def print_dataset_info(self):
        print('------------DatasetProcessor Info------------')
        print(f'Number of images: {self.num_of_samples}')
        print(f'Number of training images: {self.num_of_train_images}')
        print(f'Number of validaiton images: {self.num_of_validation_images}')
        print(f'Number of test images: {self.num_of_samples - self.num_of_train_images - self.num_of_validation_images}')
        print(f'Number of captions: {self.num_of_captions}')
        print(f'Number of vocabulary: {self.num_of_vocab}')
        print(f'Length of tallest sequence: {self.max_seq_length}')