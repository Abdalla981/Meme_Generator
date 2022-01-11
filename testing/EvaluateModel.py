import os
from keras.engine import training
import numpy as np
from typing import Tuple
from keras.preprocessing.sequence import pad_sequences
from nltk.translate.bleu_score import corpus_bleu
from training.MergeModel import MergeModel
from utils.MemeDrawer import ImageText
from training.Dataset import Dataset

'''
This class uses a model object and path to actual images to evaluate that model and allows to 
save some sample images with generated captions.

Note: this class print the BLEU score for the generated captions eventhough it might not be the best
metric to evaluate the captions on. Also, captions are generated with greedy search instead of
beam search. The beam search will be added soon.

Methods:
- evaluate_model: this method shuffles the captions and generates a caption per image while saving 
the reference captions of that image for BLEU score later
- generate_caption: generates a caption using an image (embedding)
- show_generated_examples: shows (and saves) sample images with generated captions
- print_evaluation: prints BLEU scores
'''

class EvaluateModel():
    def __init__(self, model_obj: MergeModel, images_path: str, num: int=None) -> None:
        self.model_obj = model_obj
        self.dp_obj = model_obj.dp_obj
        self.num = num
        self.true_images = Dataset(images_path=images_path).images
        self.og_captions, self.gen_captions = self.evaluate_model()
        
    def evaluate_model(self) -> Tuple[list, dict]:
        og_captions, gen_captions = [], {}
        names = list(self.dp_obj.captions.items())
        np.random.shuffle(names)
        for name, captions in names[:self.num]:
            image = self.dp_obj.images[name]
            gen_caption = self.generate_caption(image)
            refrences = [caption.split() for caption in captions]
            og_captions.append(refrences)
            gen_captions[name] = gen_caption.split()
        return og_captions, gen_captions
            
    def generate_caption(self, image) -> str:
        in_text = self.dp_obj.start_token
        max_seq_length = self.dp_obj.max_seq_length
        image = np.expand_dims(image, 0)
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
    
    def show_generated_examples(self, num: int=5, color: Tuple[int, int, int]=(255, 255, 255), 
                                output_folder_path: dict=None) -> None:
        names = [np.random.choice(list(self.gen_captions.keys())) for i in range(num)]
        for name in names:
            caption = self.gen_captions[name]
            # split the text in half
            text1 = ' '.join(caption[1:len(caption)//2])
            text2 = ' '.join(caption[len(caption)//2:-1])
            image = self.true_images[name]
            if image is not None:
                img = ImageText(image)
                # write the first half of the text on top
                img.write_text_box(0, 0, text1, box_width=img.size[0]+2,
                                   font_size=32, color=color, place='center')
                # write the second half of the text on bottom
                img.write_text_box(0, 255, text2, box_width=img.size[0]+2,
                                   font_size=32, color=color, place='center', bottom=True)
                img.image.show()
                if output_folder_path is not None:
                    img.save(os.path.join(output_folder_path, name + '.jpg'))

        
    def print_evaluation(self) -> None:
        print('BLEU-1: %f' % corpus_bleu(self.og_captions, list(self.gen_captions.values()), weights=(1.0, 0, 0, 0)))
        print('BLEU-2: %f' % corpus_bleu(self.og_captions, list(self.gen_captions), weights=(0.5, 0.5, 0, 0)))
        print('BLEU-3: %f' % corpus_bleu(self.og_captions, list(self.gen_captions), weights=(0.3, 0.3, 0.3, 0)))
        print('BLEU-4: %f' % corpus_bleu(self.og_captions, list(self.gen_captions), weights=(0.25, 0.25, 0.25, 0.25)))
