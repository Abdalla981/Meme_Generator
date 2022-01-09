from os import sep
import numpy as np
from preprocessing.Dataset import Dataset

class Glove(Dataset):
    def __init__(self, captions_path: str, glove_path:str) -> None:
        super().__init__(captions_path=captions_path)
        self.glove_path = glove_path
        
    def load_glove(self) -> dict:
        embeddings_index = {}
        with open(self.glove_path, 'r') as f:
            for line in f:
                word, embedding = line.split(maxsplit=1)
                embedding = np.fromstring(embedding, dtype='f', sep=' ')
                embeddings_index[word] = embedding
        print("Found %s word vectors." % len(embeddings_index))
        return embeddings_index