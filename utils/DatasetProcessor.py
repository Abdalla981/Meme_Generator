from utils.Dataset import Dataset

'''
This class inherits the Dataset class and prepares the dataset to be passed throught the model.

Methods:
- captions_to_vocabulary: computes a set of the vocabulary used in the captions
- captions_to_list: returns a list of all captions
- add_tokens_to_caption: TODO
- captions_tokenizer: TODO
- create_sequences: TODO
- max_length: TODO
- load_photo_features: TODO
'''


class DatasetProcessor(Dataset):
    def __init__(self, captions_path, images_path, clean_dataset=True, captions_regex_str='[^a-zA-Z ]', min_captions=50):
        super().__init__(captions_path, images_path)
        self.vocab = self.captions_to_vocabulary()
        self.captions_list = self.captions_to_list()
        self.num_of_images = len(self.images)
        self.num_of_captions = len(self.captions_list)
        self.num_of_vocab = len(self.vocab)
    
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
        pass
    
    def captions_tokenizer(self):
        pass
    
    def create_sequences(self):
        pass
    
    def max_length(self):
        max_length = 0
        max_caption = " "
        captions_list = self.captions_to_list()
        for caption in captions_list:
            if max_length < len(caption):
                max_length = max(max_length, len(caption))
                max_caption = caption
        print(f"Biggest caption:{max_length} chars")
        print(max_caption)
        return max_length
    
    def load_photo_features(self):
        pass