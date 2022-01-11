import os
from ctypes import Array
from keras.models import load_model
from keras.utils.vis_utils import plot_model
from keras.layers import Dense, LSTM, Dropout, Input, Embedding, add
from tensorflow.keras.initializers import Constant
from keras.models import Model
from training.DatasetProcessor import DatasetProcessor

'''
This class implements a merge model for image captioning. It requires a DatasetProcessor object
to initialize the model architecture. The number of neurons in the 2 Dense layers and LSTM (256) is
arbitrary.

Methods:
- load_model_from_file: loads model from file using model_folder and model_name
- define_model_architecture: defines the model architecture using the DatasetProcessor object
- print_model_summary: prints the model summary and plot in the model folder
- save_model_to_folder: saves the model to folder
'''

class MergeModel():
    def __init__(self, model_folder: str, dp_obj: DatasetProcessor, 
                 init: bool=True, model_name: str=None) -> None:
        self.model_folder = model_folder
        self.model_name = model_name
        self.init = init
        self.dp_obj = dp_obj
        self.model = self.define_model_architecture() if init else self.load_model_from_file()
        
    def load_model_from_file(self) -> Model:
        path = os.path.join(self.model_folder, self.model_name)
        model = load_model(path)
        return model
        
    def define_model_architecture(self) -> Model:
        # image embedding encoder
        inputs1 = Input(shape=self.dp_obj.image_embedding_dims)
        ie1 = Dropout(0.5)(inputs1)
        ie2 = Dense(256, activation='relu')(ie1)
        
        # text embedding encoder
        inputs2 = Input(shape=(self.dp_obj.max_seq_length,))
        te1 = Embedding(self.dp_obj.num_of_vocab, self.dp_obj.glove_dims, 
                        embeddings_initializer=Constant(self.dp_obj.embedding_matrix), 
                        mask_zero=True)(inputs2)
        te2 = Dropout(0.5)(te1)
        te3 = LSTM(256)(te2)
        
        # decoder
        d1 = add([ie2, te3])
        d2 = Dense(256, activation='relu')(d1)
        ouputs = Dense(self.dp_obj.num_of_vocab, activation='softmax')(d2)
        
        model = Model(inputs=[inputs1, inputs2], outputs=ouputs)
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        return model

    def print_model_summary(self) -> None:
        path = os.path.join(self.model_folder, 'model.png')
        plot_model(self.model, to_file=path, show_shapes=True)
        print(self.model.summary())
        
    def save_model_to_folder(self, new_model_path: str) -> None:
        self.model.save(new_model_path, overwrite=False)