import os
from training.DatasetProcessor import DatasetProcessor
from training.MergeModel import MergeModel
from training.InjectModel import InjectModel
from training.CustomCallback import CustomCallback

'''
This script trains the model using the train and validation captions files. The training can be done
on google colab by uploading the required files to their servers.
'''

if __name__ == '__main__':
    images_path = 'embedding/efficientNet'
    train_captions_path = 'dataset/CaptionsClean3_train.txt'
    val_captions_path = 'dataset/CaptionsClean3_validation.txt'
    embeddings_path = 'embedding/glove/captionsGlove.pkl'
    model_folder_path = 'models/Test'
    batch_size = 1

    if input('Train?(y/n) ') == 'y':
        train_dp_obj = DatasetProcessor(train_captions_path, images_path, 'B0', embeddings_path)
        captions_list = train_dp_obj.captions_to_list()
        train_dp_obj.process_dataset(captions_list)
        train_dp_obj.embedding_matrix = train_dp_obj.create_embedding_matrix()
        
        if input('Print train dataset info?(y/n) ') == 'y':
            train_dp_obj.print_dataset_info()
            train_dp_obj.print_sample_of_sequences()
            
        val_dp_obj = DatasetProcessor(val_captions_path, images_path, 'B0', embeddings_path)
        val_dp_obj.process_dataset(captions_list)
        t_gen = train_dp_obj.data_generator(batch_size)
        val_gen = val_dp_obj.data_generator(batch_size)
        t_steps = (train_dp_obj.num_of_captions // batch_size) + 1
        val_steps = (val_dp_obj.num_of_captions // batch_size) + 1
        
        model_obj = MergeModel(model_folder_path, train_dp_obj, init=True)
        model_obj.print_model_summary()
        # define checkpoint callback
        file_path = 'test-model'
        path = os.path.join(model_folder_path, file_path)
        checkpoint = CustomCallback(path)
        # fit model
        model_obj.model.fit(t_gen, validation_data=val_gen, validation_steps=val_steps, 
                            epochs=10, steps_per_epoch=t_steps, verbose=1, callbacks=[checkpoint])
        