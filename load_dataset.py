from utils.DatasetPreProcessor import DatasetPreProcessor
from utils.DatasetProcessor import DatasetProcessor

'''
This script loads the dataset and pre-processes it and writes the clean dataset captions to a new file.
The new dataset captions and corresponding images are loaded and their info is printed
'''
if __name__ == '__main__':
    captions_path = 'Captions3.txt'
    images_path = 'memes3'
    new_captions_path = 'CaptionsClean3.txt'
    
    if input('Clean Dataset?(y/n) ') == 'y':
        preprocess_dataset_obj = DatasetPreProcessor(captions_path, images_path)
        preprocess_dataset_obj.clean_dataset()
        preprocess_dataset_obj.save_captions_to_file(new_captions_path)
    
    process_dataset_obj = DatasetProcessor(new_captions_path, images_path)
    process_dataset_obj.process_dataset()
    process_dataset_obj.print_dataset_info()
    if input('Show a training images sample?(y/n) ') == 'y':
        process_dataset_obj.show_samples(captions_dict=process_dataset_obj.train_captions, images_dict=process_dataset_obj.train_images)
    if input('Show a validation images sample?(y/n) ') == 'y':
        process_dataset_obj.show_samples(captions_dict=process_dataset_obj.validation_captions, images_dict=process_dataset_obj.validation_images)
    if input('Show a test images sample?(y/n) ') == 'y':
        process_dataset_obj.show_samples(captions_dict=process_dataset_obj.test_captions, images_dict=process_dataset_obj.test_images)
    
    