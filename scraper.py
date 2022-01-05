from __future__ import division
from bs4 import BeautifulSoup
import requests
import shutil
import os.path
import re

'''
This script scrappes the memegenerator.net website and downloads meme images and text into local drive
'''

images_path = 'memes3'
captions_path = 'Captions3.txt'
n_captions = 15   #this number *15 is the total number of captions per template
n_pages = 300  #this number is the total number of pages
Uerrors = 0
num_memes_downloaded = 0
regex = re.compile('[^a-zA-Z0-9 ]')

for i in range(1, n_pages):
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
    if len(imgs) < 1:
        break
    for j, img in enumerate(imgs):
        img_url = img['src']
        response = requests.get(img_url, stream=True)
        image_name = regex.sub('', img['alt'])
        name_of_file = '_'.join(image_name.split())
        for k in range(1,n_captions):
            if k == 1:
                URL = 'https://memegenerator.net' + links[j]['href']
            else:
                URL = 'https://memegenerator.net' + links[j]['href'] + '/images/popular/alltime/page/' + str(k)
            R = requests.get(URL)
            SOUP = BeautifulSoup(R.text,'html.parser')
            POSTS = SOUP.find_all(class_='generator-img')
            if (len(POSTS) < 1):
                break
            CHARS = SOUP.find_all(class_='single-generator')
            TAGS = [char.find(class_='optimized-instance-container img') for char in CHARS]
            MEMES = [tag.getText().strip().replace('\n', ' ') for tag in TAGS]
            with open(captions_path, 'a') as f:
                for meme in MEMES:
                    try:
                        f.write(f'{name_of_file} - {meme}\n')
                    except UnicodeEncodeError:
                        Uerrors += 1
                        pass
        completeName = os.path.join(images_path, name_of_file + '.jpg')
        with open(completeName,'wb') as out_file:
            shutil.copyfileobj(response.raw, out_file)
        del response
        num_memes_downloaded += 1

    if i % 10 == 0:
        print('<'+'='*10+'>')
        print(i)
        print(Uerrors)
        print('num_memes_downloaded =',num_memes_downloaded)
