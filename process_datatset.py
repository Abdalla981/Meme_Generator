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
    # preprocess_dataset_obj = DatasetPreProcessor(captions_path, images_path)
    # preprocess_dataset_obj.clean_dataset()
    # preprocess_dataset_obj.save_captions_to_file(new_captions_path)
    
    process_dataset_obj = DatasetProcessor(new_captions_path, images_path)
    print('-----------------------------------')
    print(f'Number of images: {process_dataset_obj.num_of_images}')
    print(f'Number of captions: {process_dataset_obj.num_of_captions}')
    print(f'Number of vocabulary: {process_dataset_obj.num_of_vocab}')
    print(f'Length of tallest sequence: {process_dataset_obj.max_seq_length}')
    for item in process_dataset_obj.get_captions_counter().most_common(10):
        print(item)
    process_dataset_obj.show_samples()
    
    