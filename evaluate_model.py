from training.MergeModel import MergeModel
from training.DatasetProcessor import DatasetProcessor, Dataset
from testing.EvaluateModel import EvaluateModel

'''
This script trains a merge model using the mergeModel, DatasetProcessor and EvaluateModel classes.
'''

if __name__ == '__main__':
    images_path = 'dataset/memes3'
    image_embeddings_path = 'embedding/efficientNet'
    train_captions_path = 'dataset/CaptionsClean3_train.txt'
    test_captions_path = 'dataset/CaptionsClean3_test.txt'
    embeddings_path = 'embedding/glove/captionsGlove.pkl'
    model_folder_path = 'models/BestModel/B0'
    model_name = 'Merge-model-B0-n256-ln256-tanhpTrue-ep004-loss4.717-val_loss5.467'

    if input('Evaluate Model?(y/n) ') == 'y':
        train_obj = Dataset(train_captions_path)
        captions_list = train_obj.captions_to_list()
        test_dp_obj = DatasetProcessor(test_captions_path, image_embeddings_path, 'B0', embeddings_path)
        test_dp_obj.process_dataset(captions_list)
        model_obj = MergeModel(model_folder_path, test_dp_obj, init=False, model_name=model_name + '.h5')
        eval_obj = EvaluateModel(model_obj, None, k=5)
        eval_obj.save_captions(model_name)
        eval_obj.save_BLEU_evaluation(model_name)
        