import numpy as np
from pickle import load
from ctypes import Array
from typing import List, Tuple
from math import floor
from training.Dataset import Dataset
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

'''
This class inherits the Dataset class and prepares the dataset to be passed throught the model.

Methods:
- process_dataset: prepares the dataset for training

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
                start_token: str='startseq', end_token: str='endseq', oov_token: str='[UNK]') -> None:
        super().__init__(captions_path, images_path, image_embedding, glove_path)
        assert(len(self.captions) == len(self.images))
        self.num_of_samples = len(self.captions)
        self.glove_dims = list(self.glove_embedding.values())[0].shape[0]
        self.image_embedding_dims = list(self.images.values())[0].shape
        self.start_token = start_token
        self.end_token = end_token
        self.oov_token = oov_token
        self.tokenizer, self.num_of_vocab = None, None
        self.max_seq_length, self.embedding_matrix = None, None
        
    def process_dataset(self, captions_list) -> None:
        self.tokenizer = self.captions_tokenizer(captions_list)
        self.num_of_vocab = len(self.tokenizer.word_index) + 1
        self.max_seq_length = self.tallest_seq_length(captions_list)
    
    def captions_tokenizer(self, captions_list: list) -> Tokenizer:
        tokenizer = Tokenizer(oov_token=self.oov_token)
        tokenizer.fit_on_texts(captions_list)
        return tokenizer
    
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
    
    def data_generator(self) -> Tuple[List[Array], Array]:
        while 1:
            names = list(self.captions.keys())
            np.random.shuffle(names)
            for name in names:
                captions = self.captions[name]
                image = self.images[name]
                in_image, in_sequence, out_word = self.create_sequences(captions, image)
                out_word = np.expand_dims(out_word, 1)
                yield ([in_image, in_sequence], out_word)
    
    def tallest_seq_length(self, captions_list: dict) -> int:
        return max(len(caption.split()) for caption in captions_list)
    
    def print_sample_of_sequences(self, num: int=2) -> None:
        j = 0
        inputs, outputs = next(self.data_generator())
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
        print(f'Number of captions: {self.num_of_captions}')
        print(f'Number of vocabulary: {self.num_of_vocab}')
        print(f'Dimension of glove embeddings: {self.glove_dims}')
        print(f'Dimension of image embeddings: {self.image_embedding_dims}')
        print(f'Length of tallest sequence: {self.max_seq_length}')
        