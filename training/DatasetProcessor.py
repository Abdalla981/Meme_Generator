import numpy as np
from pickle import load
from ctypes import Array
from typing import List, Tuple
from math import floor
from preprocessing.Dataset import Dataset
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
- create_sequences: creates an array of sequences (image, sequence[:i], next_word)
- data_generator: uses progressive loading to create sequences for better memory manegment
- create_embedding_matrix: creates embedding matrix for initializing the embedding layer later
- print_sample_of_sequences: prints a number of captions in sequences using the generator
- print_dataset_info: prints useful info on the loaded dataset
'''

class DatasetProcessor(Dataset):
    def __init__(self, captions_path: str, images_path: str, image_embedding: str, glove_path: str,
                 start_token: str='startseq', end_token: str='endseq', oov_token: str='[UNK]', 
                 split_value: tuple=(0.8, 0.1)) -> None:
        super().__init__(captions_path, images_path, image_embedding, glove_path)
        assert(len(self.captions) == len(self.images))
        self.num_of_samples = len(self.captions)
        self.glove_dims = list(self.glove_embedding.values())[0].shape[0]
        self.image_embedding_dims = list(self.images.values())[0].shape
        self.start_token = start_token
        self.end_token = end_token
        self.oov_token = oov_token
        self.split_value = split_value
        self.num_of_train_images = floor(self.num_of_samples * self.split_value[0])
        self.num_of_validation_images = floor(self.num_of_samples * self.split_value[1])
        self.tokenizer, self.num_of_vocab = None, None
        self.train_captions, self.train_images = None, None
        self.validation_captions, self.validation_images = None, None
        self.test_captions, self.test_images = None, None
        self.train_captions_list, self.max_seq_length = None, None
        
    def process_dataset(self) -> None:
        self.add_tokens_to_caption()
        (train, validation, test) = self.split_dataset()
        self.train_captions, self.train_images = train
        self.validation_captions, self.validation_images = validation
        self.test_captions, self.test_images = test
        self.train_captions_list = self.captions_to_list(self.train_captions)
        self.tokenizer = self.captions_tokenizer()
        self.num_of_vocab = len(self.tokenizer.word_index) + 1
        self.embedding_matrix = self.create_embedding_matrix()
        self.max_seq_length = self.tallest_seq_length(self.train_captions_list)
        
    def captions_to_list(self, captions_dict: dict) -> list:
        captions_list = []
        for captions in captions_dict.values():
            [captions_list.append(caption) for caption in captions]
        return captions_list
    
    def add_tokens_to_caption(self) -> None:
        for name, captions in self.captions.items():
            for i, caption in enumerate(captions):
                tokened = ' '.join([self.start_token] + caption.split() + [self.end_token])
                self.captions[name][i] = tokened
    
    def captions_tokenizer(self) -> Tokenizer:
        tokenizer = Tokenizer(oov_token=self.oov_token)
        tokenizer.fit_on_texts(self.train_captions_list)
        return tokenizer
    
    def split_dataset(self) -> Tuple[Tuple[dict, dict], Tuple[dict, dict], Tuple[dict, dict]]:
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
    
    def create_embedding_matrix(self) -> Array:
        embedding_matrix = np.zeros((self.num_of_vocab, self.glove_dims))
        words_mapping = self.tokenizer.word_index
        for word, index in words_mapping.items():
            embedding_vector = self.glove_embedding.get(word)
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector
        return embedding_matrix

    def create_sequences(self, captions: list, image: Array) -> Tuple[Array, Array, Array]:
        X1, X2, y = [], [], []
        for caption in captions:
            sequence = self.tokenizer.texts_to_sequences([caption])[0]
            for i in range(1, len(sequence)):
                in_sequence, out_word = sequence[:i], sequence[i]
                in_sequence = pad_sequences([in_sequence], padding='post', truncating='post', maxlen=self.max_seq_length)[0]
                out_word = to_categorical([out_word], num_classes=self.num_of_vocab)[0]
                X1.append(image)
                X2.append(in_sequence)
                y.append(out_word)
        return np.array(X1), np.array(X2), np.array(y)
    
    def data_generator(self, captions_dict: dict, images_dict: dict) -> Tuple[List[Array], Array]:
        while 1:
            names = list(captions_dict.keys())
            np.random.shuffle(names)
            for name in names:
                captions = captions_dict[name]
                image = images_dict[name]
                in_image, in_sequence, out_word = self.create_sequences(captions, image)
                out_word = np.expand_dims(out_word, 1)
                yield ([in_image, in_sequence], out_word)
    
    def tallest_seq_length(self, captions_list: dict) -> int:
        return max(len(caption.split()) for caption in captions_list)
    
    def print_sample_of_sequences(self, captions_dict: dict=None, images_dict: dict=None, num: int=2) -> None:
        if captions_dict is None:
            captions_dict = self.train_captions
        if images_dict is None:
            images_dict = self.train_images
        j = 0
        inputs, outputs = next(self.data_generator(captions_dict, images_dict))
        if num > outputs.shape[0]:
            num = outputs.shape[0]
        print('-------------Sequences sample---------------')
        print('{0:<15}'.format('Image size'), end=' | ')
        print('{0:<100}'.format('Sequences'), end=' | ')
        print('{0:<15}'.format('Out word'))
        for i in range(num):
            out_word = ''
            print('{0:<15}'.format(f'Caption {i}'))
            while out_word != self.end_token:
                print('{0:<15}'.format(f'{inputs[0][j].shape}'), end=' | ')
                seq = inputs[1][j].tolist()
                text = self.tokenizer.sequences_to_texts([seq])[0].replace(self.oov_token, '')
                print("{0:<100}".format(text.strip()), end=' | ')
                seq = [np.argmax(outputs[j])]
                out_word = self.tokenizer.sequences_to_texts([seq])[0]
                print('{0:<15}'.format(out_word))
                j += 1
        print('---------------------------')

    def print_dataset_info(self) -> None:
        print('------------DatasetProcessor Info------------')
        print(f'Number of images: {self.num_of_samples}')
        print(f'Number of training images: {self.num_of_train_images}')
        print(f'Number of validaiton images: {self.num_of_validation_images}')
        print(f'Number of test images: {self.num_of_samples - self.num_of_train_images - self.num_of_validation_images}')
        print(f'Number of captions: {self.num_of_captions}')
        print(f'Number of vocabulary: {self.num_of_vocab}')
        print(f'Dimension of glove embeddings: {self.glove_dims}')
        print(f'Dimension of image embeddings: {self.image_embedding_dims}')
        print(f'Length of tallest sequence: {self.max_seq_length}')
        