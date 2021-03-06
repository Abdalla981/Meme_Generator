import re
import string
from training.Dataset import Dataset
from nltk.corpus import words, wordnet, webtext, nps_chat, reuters
from nltk.stem import WordNetLemmatizer

'''
This class inherits the Dataset class and processes the dataset to clean it up 
and remove any undesired captions

Mehtods:
- clean_captions: cleans up the captions by removing punctuations, lowercasing the words 
and removing leading and trailing spaces.
- remove_non_english_words: removes non-english captions using a text corpora from nltk
(words, wordnet, webtext, nps_chat, reuters)
- remove_captions_with_non_standard_characters: removes captions that have non-latin letters
- remove_duplicate_captions: remove duplicated captions per image
- remove_low_captioned_images: removes images and corresponding captions if the number of captions is low
- remove_very_long_and_short_captions: removes very long and very short captions
- check_rare_words: accepts a caption and vocabulary counter and returns true if too many rare words
are present. Otherwise, it returns false.
- remove_captions_with_rare_words: removes captions with too many rare words.
- clean_dataset: calls all the above methods
'''

class CaptionsCleaning(Dataset):
    def __init__(self, captions_path:str, captions_regex_str='[^a-zA-Z ]', min_captions:int=50, 
                 min_words:int=2, max_words:int=25, word_freq_limit:int=3, rare_word_limit:int=2):
        super().__init__(captions_path=captions_path)
        self.captions_regex = re.compile(captions_regex_str)
        self.min_captions = min_captions
        self.min_words = min_words
        self.max_words = max_words
        self.word_freq_limit = word_freq_limit
        self.rare_word_limit = rare_word_limit
    
    def clean_dataset(self):
        self.clean_captions()
        self.remove_captions_with_non_standard_characters()
        self.remove_non_english_captions()
        self.remove_duplicate_captions()
        self.remove_very_long_and_short_captions()
        self.remove_captions_with_rare_words()
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
                                 if (self.min_words <= len(caption.split()) <= self.max_words) 
                                 and len(caption) > 3]
            del_captions += len(self.captions[name]) - len(accepted_captions)
            self.captions[name][:] = accepted_captions
        print(f'Removed {del_captions} captions due to length!')
        
    def remove_low_captioned_images(self):
        del_images = 0
        for name, captions in list(self.captions.items()):
            if len(captions) < self.min_captions:
                del self.captions[name]
                del_images += 1
        print(f'Removed {del_images} images due to insufficient captions!')
        
    def remove_captions_with_rare_words(self):
        del_captions = 0
        vocab_counter = self.get_vocabulary_counter(captions_dict=self.captions)
        for name, captions in self.captions.items():
            accepted_captions = [caption for caption in captions if not self.check_rare_words(caption, vocab_counter)]
            del_captions += len(self.captions[name]) - len(accepted_captions)
            self.captions[name] = accepted_captions
        print(f'Removed {del_captions} captions due to rare words!')
                
    def check_rare_words(self, caption, vocab_counter):
        rare_words = 0
        words = caption.split()
        for word in words:
            if vocab_counter[word] < self.word_freq_limit:
                rare_words += 1
        if rare_words > self.rare_word_limit:
            return True
        return False
            