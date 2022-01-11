import numpy as np
from pickle import dump
from typing import Tuple
from training.Dataset import Dataset

'''
This class loads all the glove embeddings and saves the embeddings for the vocabulary in the captions
only. This saves a lot of memory later when training the model.

Methods:
- load_glove: loads all the glove embeddings from the file
- get_vocabulary_embedding: loads the embeddings for words in the vocabulary and a list of words
without embeddings
- save_vocab_embedding_to_file: saves the vocabulary embeddings and a list of words without embeddings
'''

class Glove(Dataset):
    def __init__(self, captions_path: str, glove_path: str=None) -> None:
        super().__init__(captions_path=captions_path)
        self.glove_path = glove_path
        if glove_path is not None:
            self.all_embeddings = self.load_glove()
            self.vocab_embeddings, self.not_found_embeddings = self.get_vocabulary_embedding()
        
    def load_glove(self, glove_path: str=None) -> dict:
        if glove_path is None:
            glove_path = self.glove_path
        all_embeddings = {}
        with open(glove_path, 'r') as f:
            for line in f:
                word, embedding = line.split(maxsplit=1)
                embedding = np.fromstring(embedding, dtype='f', sep=' ')
                all_embeddings[word] = embedding
        print(f'Found {len(all_embeddings)} word vectors!')
        return all_embeddings
    
    def get_vocabulary_embedding(self) -> Tuple[dict, list]:
        vocab_embedding = {}
        not_found = []
        vocab = self.get_vocabulary_counter()
        for word in vocab.keys():
            embedding_vector = self.all_embeddings.get(word)
            if embedding_vector is not None:
                vocab_embedding[word] = embedding_vector
            else:
                not_found.append(word)
        return vocab_embedding, not_found
    
    def save_vocab_embedding_to_file(self, embeddings_path: str, not_found_words_path: str) -> None:
        with open(embeddings_path, 'wb') as f:
            dump(self.vocab_embeddings, f)
        not_found_string = '\n'.join(self.not_found_embeddings)
        with open(not_found_words_path, 'w') as f:
            f.write(not_found_string)
            