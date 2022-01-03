import os
import re
from PIL import Image
from collections import Counter
import MemeDrawer as MemeDrawer
import random

class Dataset:
    def __init__(self, captions_path, images_path, clean_dataset=True):
        self.image_path = images_path
        self.captions_path = captions_path
        self.captions = self.load_captions()
        self.images, self.not_found_images = self.load_images()
        if clean_dataset:
            self.clean_dataset()
        self.num_of_samples = len(self.images)
        
    def get_counter(self):    
        dataset_ditsribution = {name: len(self.captions[name]) for name in self.captions}
        counter = Counter(dataset_ditsribution)
        return counter
            
    def load_captions(self):
        data = {}
        if not os.path.exists(self.captions_path):
            raise FileNotFoundError(self.captions_path)
        with open(self.captions_path, 'r') as f:
            for line in f:
                line = line.strip().split('-')
                name = line[0].strip()
                if name in data:
                    try:
                        data[name].append(line[1])
                    except IndexError:
                       data[name].append(None) 
                else:
                    data[name] = []
        return data

    def load_image(self, name):
        if not os.path.exists(self.image_path):
            raise FileNotFoundError(self.image_path)
        regex = re.compile('[^a-zA-Z0-9 ]')
        image_name = regex.sub('', name)
        image_name = '-'.join(image_name.split()) + '.jpg'
        try:
            image = Image.open(os.path.join(self.image_path, image_name))
        except:
            return None, image_name
        return image, image_name

    def load_images(self):
        data = {}
        not_found = []
        for name in self.captions:
            image, image_name = self.load_image(name)
            data[name] = image
            if image is None:
                not_found.append((name, image_name))
        print(f'Could not find {len(not_found)} images!')
        return data, not_found
    
    def clean_dataset(self):
        for name, _ in self.not_found_images:
            del self.captions[name]
            del self.images[name]
            
    def show_samples(self, num=5, color=(255, 255, 255), output_file_path=None):
        names = [random.choice(list(self.captions.keys())) for i in range(num)]
        for name in names:
            caption = self.captions[name][0].split()
            text1 = ' '.join(caption[:len(caption)//2])
            text2 = ' '.join(caption[len(caption)//2:])
            image = self.images[name]
            img = MemeDrawer.ImageText(image)
            img.write_text_box(0, 0, text1, box_width=img.size[0]+2,
                               font_size=32, color=color, place='center')
            img.write_text_box(0, 255, text2, box_width=img.size[0]+2,
                            font_size=32, color=color, place='center', bottom=True)
            img.save(os.path.join(output_file_path, name + '.jpg'))
    

if __name__ == '__main__':
    dataset_obj = Dataset('CaptionsClean.txt', 'memes')
    dataset_obj.show_samples(output_file_path='Meme_drawer')