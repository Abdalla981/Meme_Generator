import os
import numpy as np
import json
from math import log2
from typing import Tuple
from tensorflow import reshape as tf_reshape
from keras.preprocessing.sequence import pad_sequences
from nltk.translate.bleu_score import corpus_bleu
from training.DatasetProcessor import DatasetProcessor
from training.MergeModel import MergeModel
from training.Dataset import Dataset

'''
This class uses a model and a datasetprocessor objects to generate captions for that dataset.
The generated captions are compared against the reference captions in the datasetprocessor and the
BLEU score is calculated along with the number of vocabulary in the generated captions.

Note: this class prints the BLEU score for the generated captions eventhough it might not be the best
metric to evaluate the captions on.

Methods:
- generate_captions: this method shuffles the captions and generates a caption per image while saving 
the reference captions of that image for BLEU score later
- beam_search: generates a caption for each image using beam search algorithm
- greedy_search: generates a caption for each image using greedy search algorithm
- save_BLEU_evaluation: prints BLEU scores and saves it to a text file
- save_captions: saves the generated captions into a json file
'''

class EvaluateModel():
    def __init__(self, model_obj: MergeModel, dp_obj: DatasetProcessor, generate: bool=False, num: int=None, k: int=5,
                 eval: str='beam', verbose: int=0) -> None:
        self.model_obj = model_obj
        self.dp_obj = model_obj.dp_obj if dp_obj is None else dp_obj
        self.verbose = verbose
        self.num = num
        self.k = k
        self.eval = eval
        if eval != 'beam' and eval != 'greedy':
            raise ValueError(f'eval parameter should be either "greedy" or "beam"!')
        if generate:
            self.og_captions, self.gen_captions = self.generate_captions()
        
    def generate_captions(self) -> Tuple[list, dict]:
        og_captions, gen_captions = [], {}
        names = list(self.dp_obj.captions.items())
        np.random.shuffle(names)
        for i, (name, captions) in enumerate(names[:self.num]):
            image = self.dp_obj.images[name]
            if self.eval == 'beam':
                gen_c = self.beam_search(image)
            else:
                gen_c = self.greedy_search(image)
            refrences = [caption.split() for caption in captions]
            og_captions.append(refrences)
            gen_captions[name] = gen_c.split()
            if self.verbose:
                print(f'{i}. {name}')
        return og_captions, gen_captions
            
    def greedy_search(self, image) -> str:
        in_text = self.dp_obj.start_token
        max_seq_length = self.dp_obj.max_seq_length
        for i in range(max_seq_length):
            sequence = self.dp_obj.tokenizer.texts_to_sequences([in_text])[0]
            sequence = pad_sequences([sequence], padding='post', truncating='post', maxlen=max_seq_length)[0]
            sequence = np.expand_dims(sequence, 0)
            y = self.model_obj.model([image, sequence], training=False)
            y = np.argmax(y)
            word = self.dp_obj.tokenizer.index_word.get(y)
            if word is None:
                break
            in_text += ' ' + word
            if word == self.dp_obj.end_token:
                break
        return in_text
    
    def beam_search(self, image) -> Tuple[str, int]:
        sequences = [(self.dp_obj.start_token, 0.0)]
        max_seq = self.dp_obj.max_seq_length
        end = 0
        while end < self.k:
            candidates = []
            end = 0
            for caption, score in sequences:
                if caption.split()[-1] == self.dp_obj.end_token or len(caption.split()) >= max_seq:    # if caption reached end
                    candidates.append((caption, score))
                    end += 1
                else:
                    seq = self.dp_obj.tokenizer.texts_to_sequences([caption])[0]
                    seq = pad_sequences([seq], padding='post', truncating='post', maxlen=max_seq)[0]
                    seq = seq.reshape((1, -1))
                    y = self.model_obj.model([image, seq], training=False)
                    y = tf_reshape(y, [-1])
                    best_probs = np.argsort(y)[-self.k:]
                    for index in best_probs:
                        prob = y[index]
                        if index == 0 or prob == 0:  # ignore padding and prob of 0
                            continue
                        word = self.dp_obj.tokenizer.index_word.get(index)
                        candidates.append((caption + ' ' + word, score + log2(prob)))
            sequences = sorted(candidates, key=lambda tup:tup[1])[-self.k:]
        return sequences[-1][0]
        
    def save_BLEU_evaluation(self, name) -> None:
        b1 = corpus_bleu(self.og_captions, list(self.gen_captions.values()), weights=(1.0, 0, 0, 0))
        b2 = corpus_bleu(self.og_captions, list(self.gen_captions.values()), weights=(0.5, 0.5, 0, 0))
        b3 = corpus_bleu(self.og_captions, list(self.gen_captions.values()), weights=(0.3, 0.3, 0.3, 0))
        b4 = corpus_bleu(self.og_captions, list(self.gen_captions.values()), weights=(0.25, 0.25, 0.25, 0.25))
        print(self.dp_obj.get_vocabulary_counter(self.gen_captions))
        vocab = len(self.dp_obj.get_vocabulary_counter(self.gen_captions))
        result_str = f'BLEU-1: {b1}\nBLEU-2: {b2}\nBLEU-3: {b3}\nBLEU-4: {b4}\nNumber of Vocabulary: {vocab}\n'
        print(result_str)
        f_path = os.path.join(self.model_obj.model_folder, name + 'BLEU_result.txt')
        with open(f_path, 'w') as f:
            f.write(result_str)
    
    def save_captions(self, name) -> None:
        f_path = os.path.join(self.model_obj.model_folder, name + 'generated_captions.json')
        with open(f_path, 'w') as f:
            json.dump(self.gen_captions, f)
            