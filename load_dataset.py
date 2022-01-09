import os
from preprocessing.CaptionsCleaning import CaptionsCleaning
from training.DatasetProcessor import DatasetProcessor
from model.MergeModel import MergeModel
from keras.callbacks import ModelCheckpoint

'''
This script loads the dataset and pre-processes it and writes the clean captions captions to a new file.
The new dataset is loaded and its info is printed.
'''
if __name__ == '__main__':
    captions_path = 'dataset/Captions3.txt'
    images_path = 'embedding/efficientNet'
    new_captions_path = 'dataset/CaptionsClean3.txt'
    embeddings_path = 'embedding/glove/captionsGlove.pkl'
    model_folder_path = 'model/mergeModel'
    
    if input('Clean Dataset?(y/n) ') == 'y':
        clean_obj = CaptionsCleaning(captions_path)
        clean_obj.clean_dataset()
        clean_obj.save_captions_to_file(new_captions_path)
    
    if input('Load dataset?(y/n) ') == 'y':
        process_obj = DatasetProcessor(new_captions_path, images_path, 'B0', embeddings_path)
        process_obj.process_dataset()
        process_obj.print_dataset_info()
        # process_obj.print_sample_of_sequences()
        
        if input('Train?(y/n) ') == 'y':
            model_obj = MergeModel(model_folder_path, process_obj, init=True)  
            generator = process_obj.data_generator(process_obj.train_captions, process_obj.train_images)
            # define checkpoint callback
            file_path = 'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
            path = os.path.join(model_folder_path, file_path)
            checkpoint = ModelCheckpoint(path, monitor='loss', verbose=1, save_best_only=True, mode='min')
            # fit model
            steps = process_obj.num_of_samples
            model_obj.model.fit(generator, epochs=10, steps_per_epoch=steps, verbose=1, callbacks=[checkpoint])