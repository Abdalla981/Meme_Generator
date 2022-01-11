import os
from preprocessing.CaptionsSplitter import CaptionsSplitter

'''
This script adds tokens and splits the captions into train, validation and test sets.
'''

if __name__ == '__main__':
    captions_path = 'dataset/CaptionsClean3.txt'
    train_captions_path = 'dataset/CaptionsClean3_train.txt'
    val_captions_path = 'dataset/CaptionsClean3_validation.txt'
    test_captions_path = 'dataset/CaptionsClean3_test.txt'
    
    if input('Split Dataset?(y/n) ') == 'y':
        splitter_obj = CaptionsSplitter(captions_path)
        splitter_obj.add_tokens_to_caption()
        train_captions, val_captions, test_captions = splitter_obj.split_dataset()
        splitter_obj.print_captions_info()
        splitter_obj.save_captions_to_file(train_captions_path, train_captions)
        splitter_obj.save_captions_to_file(val_captions_path, val_captions)
        splitter_obj.save_captions_to_file(test_captions_path, test_captions)
        