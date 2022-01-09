from array import array
import os
import numpy as np
from pickle import dump
from typing import Tuple, Dict
from keras.layers import Input
from keras.applications.efficientnet import preprocess_input, EfficientNetB0, EfficientNetB1
from keras.applications.efficientnet import EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB5
from keras.applications.efficientnet import EfficientNetB6, EfficientNetB7
from keras.preprocessing.image import load_img, img_to_array

class EfficientNet():
    models_mapping = {
        'B0': [EfficientNetB0, 224],
        'B1': [EfficientNetB1, 240],
        'B2': [EfficientNetB2, 260],
        'B3': [EfficientNetB3, 300],
        'B4': [EfficientNetB4, 380],
        'B5': [EfficientNetB5, 456],
        'B6': [EfficientNetB6, 528],
        'B7': [EfficientNetB7, 600]
        }
    def __init__(self, images_dir:str, output_dir:str='pickle', models_list:Tuple[str,...]=('B0',)) -> None:
        self.images_dir = images_dir
        self.output_dir = output_dir
        self.models = self.init_models(models_list)
        self.features = {name:{} for name in self.models}
        
    def extract_features(self) -> Dict[str, Dict[str, array]]:
        images = [image for image in os.listdir(self.images_dir) if image.endswith('.jpg')]
        for i, image_file in enumerate(images):
            image_path = os.path.join(self.images_dir, image_file)
            image_name = image_file.split('.')[0]
            for name, model in self.models.items():
                image = load_img(image_path, target_size=(model[1], model[1]))
                image = img_to_array(image)
                image = np.expand_dims(image, axis=0)
                image = preprocess_input(image)
                feature = model[0].predict(image, verbose=0)
                self.features[name][image_name] = feature
            print(f'{i}. {image_name} done!')
          
    def init_models(self, models_list:tuple) -> dict:
        init_models = {}
        for model_name in models_list:
            model = EfficientNet.models_mapping[model_name][0]
            size = EfficientNet.models_mapping[model_name][1]
            input_layer = Input(shape=(size, size, 3))
            init_models[model_name] = [model(include_top=False, pooling='avg', input_tensor=input_layer), size]
        return init_models
    
    def save_features_to_file(self, output_dir:str=None) -> None:
        if output_dir is None:
            output_dir = self.output_dir
        for name, features in self.features.items():
            features_path = os.path.join(output_dir, name + '.pkl')
            with open(features_path, 'wb') as f:
                dump(features, f)
                