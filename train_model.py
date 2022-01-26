import os
from training.DatasetProcessor import DatasetProcessor
from training.MergeModel import MergeModel
from training.CustomCallback import CustomCallback
from testing.EvaluateModel import EvaluateModel

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

    batch_size = 128
    i = 0
    done = 8
    embeddings = ['B0']
    tanhp = [True, False]
    neurons = [128, 256, 512]
    lstm_neurons = [128, 256]
    for emb in embeddings:
        for t in tanhp:
            for n in neurons:
                for ln in lstm_neurons:
                    i += 1
                    if i < done:
                        continue
                    file_name = 'Merge-model-' + emb + '-n' + str(n) +'-ln' + str(ln) + '-tanhp' + str(t)
                    print(f'Training {file_name}...')
                    train_dp_obj = DatasetProcessor(train_captions_path, images_path, emb, embeddings_path)
                    captions_list = train_dp_obj.captions_to_list()
                    train_dp_obj.process_dataset(captions_list)
                    train_dp_obj.embedding_matrix = train_dp_obj.create_embedding_matrix()
                    val_dp_obj = DatasetProcessor(val_captions_path, images_path, emb, embeddings_path)
                    val_dp_obj.process_dataset(captions_list)
                    t_gen = train_dp_obj.data_generator(batch_size)
                    val_gen = val_dp_obj.data_generator(batch_size)
                    t_steps = (train_dp_obj.num_of_captions // batch_size) + 1
                    val_steps = (val_dp_obj.num_of_captions // batch_size) + 1
                    # Load Model
                    model_obj = MergeModel(model_folder_path, train_dp_obj, init=True, neurons=n, lstm_neurons=ln, tanhp=t)
                    # define file path
                    path = os.path.join(model_obj.model_folder, file_name)
                    callback = CustomCallback(path, patience=1)
                    # fit model 
                    history = model_obj.model.fit(t_gen, validation_data=val_gen, validation_steps=val_steps, epochs=10, steps_per_epoch=t_steps, verbose=0, callbacks=[callback])
                    eval_obj = EvaluateModel(model_obj, None, dp_obj=val_dp_obj, k=5)
                    eval_obj.save_captions(file_name)
                    eval_obj.save_BLEU_evaluation(file_name)
                    