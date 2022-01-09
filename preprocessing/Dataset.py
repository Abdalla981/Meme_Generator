import os
import random
import utils.MemeDrawer as MemeDrawer
from collections import Counter
from PIL import Image
from typing import Tuple
from pickle import load, dump
'''
This is a dataset class to load the captions and images from file

Captions are stored as a dictionary of {image_name: list_of_captions}
Images are stored as a dictionary of {image_name: pillow_image_object}

Methods:
- get_captions_counter: returns the number of samples per image_name as a collection counter
- get_vocabulary_counter: returns a counter of all vocabulary words
- load_captions: loads the captions from txt file into a dictionary
- load_image: loads image from folder using the name parameter and image_path variable
- load_images: iterates over captions to load the correpsponding images using load_image method
- load_image_embeddings: loads the image embeddings and returns only the embeddings of images that have caption
- load_glove_embedding: loads the glove embeddings dictionary from file
- show_samples: randomly selects images and writes a corresponding caption on them using MemeDrawer.py
- save_captions_to_file: saves the captions to a new file
- save_images_to_folder: saves the images to a new folder
- save_image_features_to_file: saves image embeddings to a file
'''

class Dataset:
    def __init__(self, captions_path: str=None, images_path: str=None, image_embedding: str=None, 
                 glove_path: str=None) -> None:
        self.image_path = images_path
        self.captions_path = captions_path
        self.image_embedding = image_embedding
        self.glove_path = glove_path
        self.captions, self.num_of_captions = None, None
        self.images, self.not_found_images = None, None
        self.glove_embedding = None
        if captions_path is not None:
            self.captions, self.num_of_captions = self.load_captions()
        if images_path is not None:
            self.images, self.not_found_images = self.load_images() if image_embedding is None else self.load_image_embeddings()
        if glove_path is not None:
            self.glove_embedding = self.load_glove_embedding(glove_path)
        
    def load_captions(self) -> Tuple[dict, int]:
        data = {}
        num_of_captions = 0
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
                num_of_captions += 1
        return data, num_of_captions

    def load_image(self, name: str) -> Tuple[Image.Image, str]:
        if not os.path.exists(self.image_path):
            raise FileNotFoundError(self.image_path)
        image_path = os.path.join(self.image_path, name + '.jpg')
        image = Image.open(image_path) if os.path.exists(image_path) else None
        return image, image_path

    def load_images(self) -> Tuple[dict, list]:
        data = {}
        not_found = []
        for name in self.captions:
            image, image_path = self.load_image(name)
            data[name] = image
            if image is None:
                not_found.append((name, image_path))
        return data, not_found
    
    def load_image_embeddings(self) -> Tuple[dict, list]:
        data = {}
        image_features = {}
        not_found = []
        file = self.image_embedding + '.pkl'
        file_path = os.path.join(self.image_path, file)
        with open(file_path, 'rb') as f:
            data = load(f)
        names = list(data.keys())
        for name in self.captions:
            if name in names:
                image_features[name] = data[name]
            else:
                not_found.append(name)
        return image_features, not_found
    
    def load_glove_embedding(self, glove_path: str) -> dict:
        glove_embeddings = {}
        with open(glove_path, 'rb') as f:
            glove_embeddings = load(f)
        return glove_embeddings
        
    def get_captions_counter(self, captions_dict: dict=None) -> Counter:
        if captions_dict is None:
            captions_dict = self.captions
        dataset_ditsribution = {
            name: len(captions) for name, captions in captions_dict.items()}
        counter = Counter(dataset_ditsribution)
        return counter
    
    def get_vocabulary_counter(self, captions_dict: dict=None) -> Counter:
        if captions_dict is None:
            captions_dict = self.captions
        words = []
        [words.extend(caption.split()) 
         for captions in captions_dict.values() for caption in captions]
        vocab = Counter(words)
        return vocab

    def show_samples(self, images_dict: dict=None, captions_dict: dict=None, num: int=5, 
                     color: Tuple[int, int, int]=(255, 255, 255), output_file_path: dict=None) -> None:
        if images_dict is None:
            images_dict = self.images
        if captions_dict is None:
            captions_dict = self.captions
        names = [random.choice(list(captions_dict.keys())) for i in range(num)]
        for name in names:
            caption = captions_dict[name][0].split()
            # split the text in half
            text1 = ' '.join(caption[:len(caption)//2])
            text2 = ' '.join(caption[len(caption)//2:])
            image = images_dict[name]
            if image is not None:
                img = MemeDrawer.ImageText(image)
                # write the first half of the text on top
                img.write_text_box(0, 0, text1, box_width=img.size[0]+2,
                                   font_size=32, color=color, place='center')
                # write the second half of the text on bottom
                img.write_text_box(0, 255, text2, box_width=img.size[0]+2,
                                   font_size=32, color=color, place='center', bottom=True)
                img.image.show()
                if output_file_path is not None:
                    img.save(os.path.join(output_file_path, name + '.jpg'))

    def save_captions_to_file(self, new_captions_path: str, captions_dict: dict=None) -> None:
        if new_captions_path == self.captions_path:
            raise ValueError('Can not write to same file!')
        if captions_dict is None:
            captions_dict = self.captions
        caption_file = open(new_captions_path, 'w')
        for name, captions in captions_dict.items():
            for caption in captions:
                caption_file.write(f'{name} - {caption}\n')
        caption_file.close()
        
    def save_images_to_folder(self, new_images_path: str, images_dict: dict=None) -> None:
        if new_images_path == self.image_path:
            raise ValueError('Can not write to same folder!')
        if images_dict is None:
            images_dict = self.images
        for name, image in images_dict.item():
            image.save(os.path.join(new_images_path, name + '.jpg'))
            
    def save_image_features_to_file(self, new_images_path:str, images_dict:dict=None) -> None:
        if new_images_path == self.image_path:
            raise ValueError('Can not write to same file!')
        if images_dict is None:
            images_dict = self.images
        features_path = os.path.join(new_images_path, self.image_embedding + '-dataset.pkl')
        with open(features_path, 'wb') as f:
            dump(images_dict, f)
            