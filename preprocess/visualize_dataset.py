import h5py
import ipdb
import json
import os
import glob
import numpy as np
import pandas as pd
import scipy.io
from collections import defaultdict
import ipdb
from itertools import chain

'''
    UTILITIES
'''

forbidden_verbs = ['said']

def convertTime(value):
    second = value%60
    minute = int(value/60)%60
    hour = int(value/3600)
    return '{}:{:02}:{:02}'.format(hour, minute, second)

def getBookName(filen):
    filen.split('/')[-1]
    book_path = '{}/books/'.format(bksnmvs_path)
    book_name = filen+'.(hl-2).mat'
    return book_path+book_name


'''
    PATHS AND NAMES
'''

dataset_path = '/data/vision/torralba/datasets/movies/data/'
bksnmvs_path = '/data/vision/torralba/frames/data_acquisition/booksmovies/data/booksandmovies/'
anno_path = '{}/antonio/annotation/'.format(bksnmvs_path)
text_annotation_path = '/data/vision/torralba/movies-books/booksandmovies/joanna/bksnmovies/data/gt_alignment/consecutive_text_labels_v2'
labeled_sents = json.load(open('../data/bksnmovies/labeled_sents.json', 'r'))
movies = ['American.Psycho','Brokeback.Mountain','Fight.Club','Gone.Girl','Harry.Potter.and.the.Sorcerers.Stone','No.Country.for.Old.Men','One.Flew.Over.the.Cuckoo.Nest','Shawshank.Redemption','The.Firm','The.Green.Mile','The.Road']
movies_titles = [movie.replace('.', '_') for movie in movies]
movies_titles[7] = 'The_Shawshank_Redemption'
movie_to_title = {movie:title for movie, title in zip(movies, movies_titles)}

imbds = ['tt0144084','tt0388795','tt0137523','tt2267998','tt0241527','tt0477348','tt0073486','tt0111161','tt0106918','tt0120689','tt0898367']
dataset_split = 1 # 1 for 90% data from all movies in train and 10% in val; 2 for n-1 movies in train 1 in val
val_movie = movies[0]

text_data = json.load(open('../annotations/bksnmovies/text_data.json', 'r'))
vid_data = json.load(open('../annotations/bksnmovies/vid_data.json', 'r'))
pos_data = json.load(open('../annotations/bksnmovies/pos_data.json', 'r'))

coot2 = json.load(open('../annotations/bksnmovies/bknmovies_v0_split_{}_nonempty.json'.format(dataset_split), 'r'))

lines = ["<html>",
        "<h2>Harry Potter</h2>",
        "<table>",
        "<tbody>"
]

counter = 0

# Get alignment pairs
for key, gt in coot2['database'].items():
    movie = key.split('_')[0]
    if counter < 20 and movie == 'Harry.Potter.and.the.Sorcerers.Stone':
        # Get video
        title = movie_to_title[movie]
        # Get mp4
        print('{}/movies/{}/*.mp4'.format(dataset_path, title))
        vid_movie = glob.glob('{}/movies/{}/*.mp4'.format(dataset_path, title))
        if vid_movie == []: vid_movie = glob.glob('{}/movies/{}/*.m4v'.format(dataset_path, title))
        if vid_movie == []: vid_movie = glob.glob('{}/movies/{}/*.avi'.format(dataset_path, title))
        vid_movie = vid_movie[0]
        print('vid_movie is: {}'.format(vid_movie))

        # Get text
        if key in text_data.keys():
            text_dict = text_data[key]
            text = ' '.join([text for id_sent, text in text_dict.items()])
            # Get scene start
            scene_start = gt['scene_start']
            
            lines.append('<tr>')
            lines.append('<table><theader> <h3>Alignment {} </h3></theader>'.format(key))
            lines.append('<tbody>')
            lines.append('<tr>')
            # Get shots
            vid_dict = vid_data[key]
            for id_shot, shot in vid_dict.items():
                clip_start = scene_start + shot[0]
                clip_end = scene_start + shot[1]
                clip_dur = clip_end - clip_start
                
                out_name = '~/public_html/tmp2/{}_{}.mp4'.format(key, id_shot)
                print(out_name)
                print(f"clip_start: {clip_start}, clip_dur: {clip_dur}")
                cmd =  'ffmpeg -y -nostats -loglevel 0 -ss {} -i {} -t {} -vcodec libx264 -acodec aac -strict -2 {}'.format(clip_start, vid_movie, clip_dur, out_name)
                os.system(cmd)
                
                
                src_name = 'tmp2/{}_{}.mp4'.format(key,id_shot)
                lines.append('<td><video preload="none" width="500px"; controls><source src="{}"></video></td>'.format(src_name))
            
            lines.append('</tr>')
            lines.append('<tr> <td>{} </td></tr>'.format(text))
            
            '''
            for id_sent, sent in text_dict.items():
                lines.append('<div style=color:black> {}: {}</div>'.format(id_sent, sent))
            '''
            
            for k, bag in enumerate(pos_data[key]):
                anno_num = bag['anno_num']
                try:
                    lines.append('<tr><td>Bag number {} (shot {} - sent {})</td></tr>'.format(k, gt['annotations'][anno_num]['id_shot'], gt['annotations'][anno_num]['id_sentence']))
                except:
                    ipdb.set_trace()
                pos_shots = bag['positive_shots']
                pos_sents = bag['positive_sentences']
                lines.append('<tr>')
                for pos_shot in pos_shots:
                    src_name = 'tmp2/{}_{}.mp4'.format(key,pos_shot)
                    lines.append('<td><div> shot_id: {} </div><div><video preload="none" width="500px"; controls><source src="{}"></video></div>'.format(pos_shot, src_name))
                    lines.append
                    for pos_sent in pos_sents:
                        lines.append('<div>{}:  {} </div>'.format(pos_sent, text_dict[str(pos_sent)]))
                    lines.append('</td>')
                lines.append('</tr>')
            
            lines.append('</tbody></table></tr>')
            counter += 1



lines.append('</tr>')
lines.extend(['</tbody>', '</table>'])
ll = []
for i in lines:
    ll.append(i+ '\n')

with open('visualize_dataset2.html', 'w') as out:
    out.writelines(ll)
    
