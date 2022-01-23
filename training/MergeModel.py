import os
from keras.models import load_model
from keras.utils.vis_utils import plot_model
from keras.layers import Dense, LSTM, Dropout, Input, Embedding, LayerNormalization, add, Lambda
from keras.activations import tanh
from tensorflow.keras.initializers import Constant, RandomNormal
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
                 init: bool=True, model_name: str=None, activation: str=None, neurons: int=512,
                lstm_neurons: int=256, dropout: int=0, lstm_dropout: int=0, im_norm: bool=False,
                tanhp: bool=False) -> None:
        self.model_folder = model_folder
        self.model_name = model_name
        self.init = init
        self.dp_obj = dp_obj
        self.im_norm = im_norm
        self.activation = activation
        self.neurons = neurons
        self.lstm_neurons = lstm_neurons
        self.dropout = dropout
        self.lstm_dropout = lstm_dropout
        self.tanhp = tanhp
        self.model = self.define_model_architecture() if init else self.load_model_from_file()
        
    def load_model_from_file(self) -> Model:
        path = os.path.join(self.model_folder, self.model_name)
        model = load_model(path)
        return model
        
    def define_model_architecture(self) -> Model:
        # image embedding encoder
        inputs1 = Input(shape=self.dp_obj.image_embedding_dims, name='Image Input')
        ie0 = LayerNormalization(name='Image_Normalization')(inputs1) if self.im_norm else inputs1
        ie1 = Dropout(self.dropout, name='Image_Dropout')(ie0) if self.dropout > 0 else ie0
        
        # text embedding encoder
        inputs2 = Input(shape=(self.dp_obj.max_seq_length,), name='Sequence Input')
        te1 = Embedding(self.dp_obj.num_of_vocab, self.dp_obj.glove_dims, 
                        embeddings_initializer=Constant(self.dp_obj.embedding_matrix), 
                        mask_zero=True, name='Sequence_Embedding')(inputs2)
        te2 = Dropout(self.dropout, name='Embedding_Dropout')(te1) if self.dropout > 0 else te1
        te3 = LSTM(self.lstm_neurons, dropout=self.lstm_dropout, name='Sequence_Encoder',
                   kernel_initializer=RandomNormal(mean=0, stddev=0.03))(te2)
        
        # multimodal layer
        mm1 = Dense(self.neurons, kernel_initializer=RandomNormal(mean=0, stddev=0.03),
                    activation=self.activation, name='Image_Projection')(ie1)
        mm2 = Dense(self.neurons, activation=self.activation, name='Sequence_Projection')(te3)
        # mm3 = Dense(self.neurons, activation=self.activation, name='Embedding_Projection')(te1)
        d1 = add([mm1, mm2], name='Multimodal_Addition')
        d1_1 = Lambda(lambda x: x * (2/3), name='Addition_Scaling')(d1) if self.tanhp else d1
        d2 = tanh(d1_1)
        d2_1 = Lambda(lambda x: x * 1.7159, name='Tanh_Scaling')(d2) if self.tanhp else d2
        ouputs = Dense(self.dp_obj.num_of_vocab, activation='softmax', name='Softmax_Output')(d2_1)
        
        model = Model(inputs=[inputs1, inputs2], outputs=ouputs)
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        return model

    def print_model_summary(self) -> None:
        path = os.path.join(self.model_folder, 'architecture.png')
        print(self.model.summary())
        plot_model(self.model, to_file=path, show_shapes=True)
        
    def save_model_to_folder(self, new_model_path: str) -> None:
        self.model.save(new_model_path, overwrite=False)