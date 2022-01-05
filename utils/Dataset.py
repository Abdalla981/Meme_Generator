import os
import re
import random
import string
import utils.MemeDrawer as MemeDrawer
from PIL import Image
from collections import Counter


class Dataset:
    def __init__(self, captions_path, images_path, clean_dataset=True, 
                 captions_regex_str='[^a-zA-Z0-9 ?!.,\']', min_captions=50):
        self.image_path = images_path
        self.captions_path = captions_path
        self.captions_regex = re.compile(captions_regex_str)
        self.min_captions = min_captions
        self.captions = self.load_captions()
        self.images, self.not_found_images = self.load_images()
        self.vocab = self.captions_to_vocabulary()
        if clean_dataset:
            self.remove_duplicate_captions()
            self.remove_non_found_images()
            self.remove_non_standard_captions()
            self.remove_low_captioned_images()
            self.clean_dataset()
        assert(len(self.images) == len(self.captions))
        self.num_of_samples = len(self.images)

    def get_captions_counter(self):
        dataset_ditsribution = {
            name: len(captions) for name, captions in self.captions.items()}
        counter = Counter(dataset_ditsribution)
        return counter
    
    def get_vocab_counter(self):
        return Counter(self.vocab)

    def load_captions(self):
        data = {}
        num_lines = 0
        if not os.path.exists(self.captions_path):
            raise FileNotFoundError(self.captions_path)
        with open(self.captions_path, 'r') as f:
            for line in f:
                line = line.split(' - ')
                if len(line) < 2:
                    continue
                name = line[0].strip()
                if name not in data:
                    data[name] = []
                data[name].append(' '.join(line[1:]).strip())
                num_lines += 1
        return data

    def load_image(self, name):
        if not os.path.exists(self.image_path):
            raise FileNotFoundError(self.image_path)
        image_path = os.path.join(self.image_path, name + '.jpg')
        image = Image.open(image_path) if os.path.exists(image_path) else None
        return image

    def load_images(self):
        data = {}
        not_found = []
        for name in self.captions:
            image, image_name = self.load_image(name)
            data[name] = image
            if image is None:
                not_found.append((name, image_name))
        return data, not_found

    def show_samples(self, num=5, color=(255, 255, 255), output_file_path=None):
        names = [random.choice(list(self.captions.keys())) for i in range(num)]
        for name in names:
            caption = self.captions[name][0].split()
            text1 = ' '.join(caption[:len(caption)//2])
            text2 = ' '.join(caption[len(caption)//2:])
            image = self.images[name]
            if image is not None:
                img = MemeDrawer.ImageText(image)
                img.write_text_box(0, 0, text1, box_width=img.size[0]+2,
                                   font_size=32, color=color, place='center')
                img.write_text_box(0, 255, text2, box_width=img.size[0]+2,
                                   font_size=32, color=color, place='center', bottom=True)
                img.save(os.path.join(output_file_path, name + '.jpg'))
    
    def clean_dataset(self):
        table = str.maketrans('', '', string.punctuation)
        for name, captions in self.captions.items():
            for i, caption in enumerate(captions):
                # change the caption to lower case
                caption = caption.lower()
                # remove punctuations
                caption = caption.translate(table)
                # remove one character words and word with numbers in them
                caption = [word for word in caption if len(word) > 1 and word.isalpha()]
                captions[i] = ' '.join(caption)
                
    def remove_non_found_images(self):
        for name, _ in self.not_found_images:
            del self.captions[name]
            del self.images[name]
        print(f'Removed {len(self.not_found_images)} due to missing image!')

    def remove_non_standard_captions(self):
        del_captions = 0
        for name, captions in list(self.captions.items()):
            accepted_captions = [caption for caption in captions
                                      if self.captions_regex.search(caption) is None]
            del_captions += len(self.captions[name]) - len(accepted_captions)
            self.captions[name][:] = accepted_captions
        print(f'Removed {del_captions} due to non-standard format!')
        
    def remove_duplicate_captions(self):
        del_captions = 0
        for name, captions in self.captions:
            accepted_captions = list(set(captions))
            del_captions += len(self.captions[name]) - len(accepted_captions)
            self.captions[name][:] = accepted_captions
        print(f'Removed {del_captions} due to duplication!')
        
    def remove_low_captioned_images(self):
        del_images = 0
        for name, captions in list(self.captions.items()):
            if len(captions) < self.min_captions:
                del self.captions[name]
                del self.images[name]
                del_images += 1
        print(f'Removed {del_images} due to insufficient captions!')
    
    def captions_to_vocabulary(self):
        vocab = set()
        for captions in self.captions.values():
            [vocab.update(caption.split()) for caption in captions]
        return vocab
            
    def save_captions_to_file(self, new_captions_path):
        caption_file = open(new_captions_path, 'w')
        for name, captions in self.captions.items():
            for caption in captions:
                caption_file.write(f'{name} - {caption}')
        caption_file.close()
        
    def save_images_to_folder(self, new_images_path):
        for name, image in self.images.item():
            image.save(os.path.join(new_images_path, name + '.jpg'))