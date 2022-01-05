import re
import string
from utils.Dataset import Dataset
from nltk.corpus import words, wordnet, webtext, nps_chat, reuters
from nltk.stem import WordNetLemmatizer

'''
This class inherits the Dataset class and processes the dataset to clean it up 
and remove any undesired captions or images

Mehtods:
- clean_captions: cleans up the captions by removing punctuations, lowercasing the words 
and removing leading and trailing spaces.
- remove_non_english_words: removes non-english captions using a text corpus from nltk
(words, wordnet, webtext, nps_chat, reuters)
- remove_missing_images: removes images corresponding captions if the image is not found
- remove_captions_with_non_standard_characters: removes captions that have non-lating letter
- remove_duplicate_captions: remove duplicated captions per image
- remove_low_captioned_images: removes images and corresponding captions if the number of captions is low
- remove_very_long_and_short_captions: removes very long and very short captions
- clean_dataset: calls all the above methods
'''

class DatasetPreProcessor(Dataset):
    def __init__(self, captions_path, images_path, captions_regex_str='[^a-zA-Z ]', 
                 min_captions=50, min_length=2, max_length=25):
        super().__init__(captions_path, images_path)
        self.captions_regex = re.compile(captions_regex_str)
        self.min_captions = min_captions
        self.min_length = min_length
        self.max_length = max_length
    
    def clean_dataset(self):
        self.remove_missing_images()
        self.clean_captions()
        self.remove_captions_with_non_standard_characters()
        self.remove_non_english_captions()
        self.remove_duplicate_captions()
        self.remove_very_long_and_short_captions()
        self.remove_low_captioned_images()
        
    def clean_captions(self):
        table = str.maketrans('', '', string.punctuation)
        for name, captions in self.captions.items():
            for i, caption in enumerate(captions):
                # change the caption to lower case
                captioned = caption.lower()
                # remove punctuations
                captioned = captioned.translate(table)
                # remove leading and trailing white spaces 
                captioned = [word.strip() for word in captioned.split()]
                self.captions[name][i] = ' '.join(captioned)
                
    def remove_non_english_captions(self):
        del_captions = 0
        en_dict = list(words.words()) + list(wordnet.words()) + list(webtext.words())
        en_dict += list(nps_chat.words()) + list(reuters.words())
        en_words = set(w.lower() for w in en_dict)
        lemmatizer = WordNetLemmatizer().lemmatize
        lemm_func = lambda x: lemmatizer(x)
        for name, captions in self.captions.items():
            english_captions = [caption for caption in captions 
                                if (len(set(lemm_func(word) for word in caption.split()) - en_words) < 1)]
            del_captions += len(self.captions[name]) - len(english_captions)
            self.captions[name][:] = english_captions
        print(f'Removed {del_captions} captions due to non-english words!')
                
    def remove_missing_images(self):
        for name, _ in self.not_found_images:
            del self.captions[name]
            del self.images[name]
        print(f'Removed {len(self.not_found_images)} images due to missing file!')

    def remove_captions_with_non_standard_characters(self):
        del_captions = 0
        for name, captions in self.captions.items():
            accepted_captions = [caption for caption in captions
                                      if self.captions_regex.search(caption) is None]
            del_captions += len(self.captions[name]) - len(accepted_captions)
            self.captions[name][:] = accepted_captions
        print(f'Removed {del_captions} captions due to non-standard format!')
        
    def remove_duplicate_captions(self):
        del_captions = 0
        for name, captions in self.captions.items():
            accepted_captions = list(set(captions))
            del_captions += len(self.captions[name]) - len(accepted_captions)
            self.captions[name][:] = accepted_captions
        print(f'Removed {del_captions} captions due to duplication!')

    def remove_very_long_and_short_captions(self):
        del_captions = 0
        for name, captions in self.captions.items():
            accepted_captions = [caption for caption in captions 
                                 if (self.min_length <= len(caption.split()) <= self.max_length) 
                                 and len(caption) > 3]
            del_captions += len(self.captions[name]) - len(accepted_captions)
            self.captions[name][:] = accepted_captions
        print(f'Removed {del_captions} captions due to length!')
        
    def remove_low_captioned_images(self):
        del_images = 0
        for name, captions in list(self.captions.items()):
            if len(captions) < self.min_captions:
                del self.captions[name]
                del self.images[name]
                del_images += 1
        print(f'Removed {del_images} images due to insufficient captions!')
            