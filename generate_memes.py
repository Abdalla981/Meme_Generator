from training.Dataset import Dataset
from json import load

if __name__ == '__main__':
    gen_captions_path = 'models/BestModel/B0/test_generated_captions.json'
    images_path = 'dataset/memes3'
    output_folder = 'models/BestModel/B0/samples'
    template_names = ['MrBrut', 'Gamer_rage', 'Idiot_Football_Coach', 
                      'Counter_Strike', 'Success_Rdj', 'Jimmy_Pulp_Fiction', 
                      'Musically_Diverse_Metalhead', 'Look_at_all_the_things', 'truestoryxd',
                      'Cool_Drum_Corps', 'Uneducatedchicken', 'Trans_Parrotfish']
    
    sel_captions = {}
    with open(gen_captions_path, 'r') as f:
        captions = load(f)
    for name in template_names:
        sel_captions[name] = captions[name]
    d_obj = Dataset(images_path=images_path)
    d_obj.show_samples(captions_dict=sel_captions, splitted=True, output_file_path=output_folder, 
                       num=len(template_names), random=False)