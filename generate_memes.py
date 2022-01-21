from training.Dataset import Dataset
from json import load

if __name__ == '__main__':
    gen_captions_path = 'models/BaseModel/generated_captions_B6.json'
    images_path = 'dataset/memes3'
    output_folder = 'Meme_drawer/Model_generated/BaseModel/B6'
    
    with open(gen_captions_path, 'r') as f:
        captions = load(f)
        
    d_obj = Dataset(images_path=images_path)
    d_obj.show_samples(captions_dict=captions, splitted=True, num=10, output_file_path=output_folder)