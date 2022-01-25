from training.Dataset import Dataset
from json import load
import random

if __name__ == '__main__':
    gen_captions_path = 'models/BestModel/B0/test_generated_captions.json'
    captions_path = 'dataset/CaptionsClean3_test.txt'
    images_path = 'dataset/memes3'
    output_folder = 'models/BestModel/B0/samples'
    human_folder_path = 'models/HumanSamples2'
    template_names = ['MrBrut', 'Gamer_rage', 'Idiot_Football_Coach', 
                      'Counter_Strike', 'Success_Rdj', 'Jimmy_Pulp_Fiction', 
                      'Musically_Diverse_Metalhead', 'Look_at_all_the_things', 'truestoryxd',
                      'Cool_Drum_Corps', 'Uneducatedchicken', 'Trans_Parrotfish']
    
    if input('Model?(y/n) ') == 'y':
        sel_captions = {}
        with open(gen_captions_path, 'r') as f:
            captions = load(f)
        for name in template_names:
            sel_captions[name] = captions[name]
        d_obj = Dataset(images_path=images_path, captions_path=captions_path)
        d_obj.show_samples(captions_dict=sel_captions, splitted=True, output_file_path=output_folder, 
                        num=len(template_names), random=False)
    if input('Human?(y/n) ') == 'y':
        sel_captions = {}
        d_obj = Dataset(images_path=images_path, captions_path=captions_path)
        for name in template_names:
            s_no = len(d_obj.captions[name])
            sel_captions[name] = [d_obj.captions[name][random.randint(0, s_no - 1)]]
        # print(sel_captions)
        d_obj.show_samples(captions_dict=sel_captions, splitted=False, output_file_path=human_folder_path, 
                        num=len(template_names), random=False)