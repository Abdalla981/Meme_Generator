from preprocessing.CaptionsCleaning import CaptionsCleaning

'''
This script loads the dataset and pre-processes it and 
writes the clean captions captions to a new file.
'''

if __name__ == '__main__':
    captions_path = 'dataset/Captions3.txt'
    new_captions_path = 'dataset/CaptionsClean3.txt'
    
    if input('Clean Dataset?(y/n) ') == 'y':
        clean_obj = CaptionsCleaning(captions_path)
        clean_obj.clean_dataset()
        clean_obj.save_captions_to_file(new_captions_path)
