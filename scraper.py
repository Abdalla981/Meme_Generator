from __future__ import division
from bs4 import BeautifulSoup
import requests
import shutil
import os.path

save_path = 'memes2'
n_captions = 14   #this number *15 is the total number of captions per template
n_templates = 999  #this number *15 is the total number of templates
Uerrors = 0
already_downloaded = None
num_memes_downloaded = 0

for i in range(1,n_templates):
    if i == 1:
        url = 'https://memegenerator.net/memes/popular/alltime'
        # url = 'https://memegenerator.net/memes/popular/month'
    else:
        url = 'https://memegenerator.net/memes/popular/alltime/page/' + str(i)
        # url = 'https://memegenerator.net/memes/popular/month/page/' + str(i)

    r = requests.get(url)
    soup = BeautifulSoup(r.text,'html.parser')
    chars = soup.find_all(class_='char-img')
    links = [char.find('a') for char in chars]
    imgs = [char.find('img') for char in chars]
    assert len(links) == len(imgs)
    for j,img in enumerate(imgs):

        img_url = img['src']
        response = requests.get(img_url, stream=True)
        name_of_file = img_url.split('/')[-1]

        for k in range(1,n_captions):
            if k == 1:
                URL = 'https://memegenerator.net' + links[j]['href']
            else:
                URL = 'https://memegenerator.net' + links[j]['href'] + '/images/popular/alltime/page/' + str(k)

            R = requests.get(URL)
            SOUP = BeautifulSoup(R.text,'html.parser')
            if k==1:
                #check if meme has > n_captions posts
                POSTS = SOUP.find_all(class_='generator-img')
                if (len(POSTS) < n_captions*15 or name_of_file in already_downloaded):
                    continue
            CHARS = SOUP.find_all(class_='single-generator')
            TAGS = [char.find(class_='optimized-instance-container img') for char in CHARS]
            MEMES = [tag.getText().strip().replace('\n', ' ') for tag in TAGS]
            with open('Captions2.txt', 'a') as f:
                for meme in MEMES:
                    try:
                        f.write(f'{img["alt"]} - {meme}\n')
                    except UnicodeEncodeError:
                        Uerrors += 1
                        pass

        completeName = os.path.join(save_path, name_of_file)
        with open(completeName,'wb') as out_file:
            shutil.copyfileobj(response.raw, out_file)
        del response
        num_memes_downloaded += 1

    if i % 10 == 0:
        print('<'+'='*10+'>')
        print(i)
        print(Uerrors)
        print('num_memes_downloaded =',num_memes_downloaded)
