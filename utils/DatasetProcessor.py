from utils.Dataset import Dataset

'''
This class inherits the Dataset class and prepares the dataset to be passed throught the model.

Methods:
- captions_to_vocabulary: computes a set of the vocabulary used in the captions
- captions_to_list: returns a list of all captions
- add_tokens_to_caption: Adds start and end token to each caption
- captions_tokenizer: TODO
- create_sequences: TODO
- add_tokens_to_caption: returns the number of words in the tallest caption
- load_photo_features: TODO
'''

class DatasetProcessor(Dataset):
    def __init__(self, captions_path, images_path, start_token='startseq', end_token='endseq'):
        super().__init__(captions_path, images_path)
        self.start_token = start_token
        self.end_token = end_token
        self.add_tokens_to_caption()
        self.vocab = self.captions_to_vocabulary()
        self.captions_list = self.captions_to_list()
        self.num_of_images = len(self.images)
        self.num_of_captions = len(self.captions_list)
        self.num_of_vocab = len(self.vocab)
        self.max_seq_length = self.tallest_seq_length()
    
    def captions_to_vocabulary(self):
        vocab = set()
        for captions in self.captions.values():
            [vocab.update(caption.split()) for caption in captions]
        return vocab
    
    def captions_to_list(self):
        captions_list = []
        for captions in self.captions.values():
            [captions_list.append(caption) for caption in captions]
        return captions_list
    
    def add_tokens_to_caption(self):
        for name, captions in self.captions.items():
            for i, caption in enumerate(captions):
                tokened = ' '.join([self.start_token] + caption.split() + [self.end_token])
                self.captions[name][i] = tokened
    
    def captions_tokenizer(self):
        pass
    
    def create_sequences(self):
        pass
    
    def tallest_seq_length(self):
        return max(len(caption.split()) for caption in self.captions_list)
    
    def load_photo_features(self):
        pass