import numpy as np
from math import floor
from typing import Tuple
from training.Dataset import Dataset

'''
This class implements a captions spliter to split the captions into train, validaiton and test sets.
The tokens (start and end) are added before the splitting.

Methods:
- add_tokens_to_caption: Adds start and end token to each caption
- split_dataset: splits the dataset into 3 sets (train, val, test)
- print_dataset_info: prints captions info
'''
class CaptionsSplitter(Dataset):
    def __init__(self, captions_path: str, split_value: tuple=(0.8, 0.1), 
                 start_token: str='startseq', end_token: str='endseq') -> None:
        super().__init__(captions_path=captions_path)
        self.num_of_samples = len(self.captions)
        self.start_token = start_token
        self.end_token = end_token
        self.split_value = split_value
        self.num_of_train_images = floor(self.num_of_samples * self.split_value[0])
        self.num_of_validation_images = floor(self.num_of_samples * self.split_value[1])
        self.num_of_test_images = self.num_of_samples - (self.num_of_train_images + self.num_of_validation_images)
        
    def add_tokens_to_caption(self) -> None:
        for name, captions in self.captions.items():
            for i, caption in enumerate(captions):
                tokened = ' '.join([self.start_token] + caption.split() + [self.end_token])
                self.captions[name][i] = tokened
                
    def split_dataset(self) -> Tuple[Tuple[dict, dict], Tuple[dict, dict], Tuple[dict, dict]]:
        train_captions, val_captions, test_captions = {}, {}, {}
        names = list(self.captions.keys())
        np.random.shuffle(names)
        train_index = self.num_of_train_images
        validation_index = self.num_of_train_images + self.num_of_validation_images
        for name in names[:train_index]:
            train_captions[name] = self.captions[name]
        for name in names[train_index:validation_index]:
            val_captions[name] = self.captions[name]
        for name in names[validation_index:]:
            test_captions[name] = self.captions[name]
        return train_captions, val_captions, test_captions
    
    def print_captions_info(self) -> None:
        print('------------DatasetProcessor Info------------')
        print(f'Number of images: {self.num_of_samples}')
        print(f'Number of training images: {self.num_of_train_images}')
        print(f'Number of validaiton images: {self.num_of_validation_images}')
        print(f'Number of test images: {self.num_of_test_images}')
        print(f'Number of captions: {self.num_of_captions}')
        
        