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


def isVisual(labs):
    for lab in labs:
        if 'description' in lab: return True
    return False

def getVisualSents(center, window, max_dist, labels):
    dist = {}
    for i in range(max(0, center - max_dist), min(len(labels), center + max_dist)):
        if 'visual' in labels[str(i)]: dist[i] = abs(center - i)
    # sort dist dictionary 
    dist = {k: v for k, v in sorted(dist.items(), key=lambda item: item[1])}
    neighbs = [k for k in dist.keys()][:window]
    return neighbs


def locateInVector(value, vector):
    counter = 0
    while counter < len(vector) and vector[counter] < value:
        counter += 1
    return counter


'''
    PATHS AND NAMES
'''

dataset_path = '/data/vision/torralba/datasets/movies/data/'
bksnmvs_path = '/data/vision/torralba/frames/data_acquisition/booksmovies/data/booksandmovies/'
anno_path = '{}/antonio/annotation/'.format(bksnmvs_path)
text_annotation_path = '/data/vision/torralba/movies-books/booksandmovies/joanna/bksnmovies/data/gt_alignment/consecutive_text_labels_v2'
dialog_anno = json.load(open('../data/bksnmovies/dialog_anno.json', 'r'))
labeled_sents = json.load(open('../data/bksnmovies/labeled_sents.json', 'r'))
movies = ['American.Psycho','Brokeback.Mountain','Fight.Club','Gone.Girl','Harry.Potter.and.the.Sorcerers.Stone','No.Country.for.Old.Men','One.Flew.Over.the.Cuckoo.Nest','Shawshank.Redemption','The.Firm','The.Green.Mile','The.Road']
movies_titles = [movie.replace('.', '_') for movie in movies]
imbds = ['tt0144084','tt0388795','tt0137523','tt2267998','tt0241527','tt0477348','tt0073486','tt0111161','tt0106918','tt0120689','tt0898367']
dataset_split = 1  # 1 for 90% data from all movies in train and 10% in val; 2 for n-1 movies in train 1 in val
val_movie = movies[0]

sent_window = 1
shot_window = 1

# Ground-truth annotations
coot = defaultdict(dict)
text_data = {}
vid_data = {}
pos_data = {}


total_scene_duration = 0


for i, movie in enumerate(movies):
    print('Getting ground-truth for movie: ' + movie)

    title = movies_titles[i]
    imbd = imbds[i]

    labels = dialog_anno[movie]
    ### SCENE AND SHOTS

    # Load shot info
    srt = scipy.io.loadmat('{}srts/{}.mat'.format(bksnmvs_path, movie))

    # Get scene breaks
    scene_break = srt['shot']['scene_break'][0][0] # shape (#scenes, 3)
    scene_break[:,:2] -= 1 # substract one to get actual shot id
    scene_df = pd.DataFrame(data=scene_break, columns=['start_shot','end_shot','num_shots'])
    scene_df.index.name = 'scene_id'

    # Get shots
    shot_time = srt['shot']['time'][0][0] # shape (#shots, 2)
    num_shots = shot_time.shape[0]
    shot_df = pd.DataFrame(data = shot_time, columns =['start_time', 'end_time'])
    
    # Add shot duration
    shot_df['duration'] = shot_df['end_time'] - shot_df['start_time']
    
    # Add scene id for each shot
    shot_to_scene = np.empty(num_shots, dtype='int64')
    for id_shot in range(num_shots):
        starting = (scene_df['start_shot'] <= id_shot).values
        ending = (scene_df['end_shot'] >= id_shot).values
        scene_id = np.argmax(starting & ending)
        shot_to_scene[id_shot] = scene_id
    shot_df['scene_id'] = shot_to_scene
    
  
    # Add scene start time, end time and duration to scene_df
    scene_df['scene_start_time'] = shot_df[['scene_id', 'start_time']].groupby(by='scene_id').min()
    scene_df['scene_end_time'] = shot_df[['scene_id', 'end_time']].groupby(by='scene_id').max()
    scene_df['scene_duration'] = scene_df['scene_end_time'] - scene_df['scene_start_time']
    total_scene_duration += scene_df['scene_duration'].sum()
    
    # Add scenes to COOT annotations
    for scene_id in scene_df.index:
        key = movie + '_' + str(scene_id)
        coot['database'][key] = {}
        coot['database'][key]['duration'] = scene_df['scene_duration'][scene_id]
        coot['database'][key]['scene_start'] = scene_df['scene_start_time'][scene_id]
        coot['database'][key]['annotations'] = []
        if dataset_split == 1:
            if scene_id <= len(scene_df)*.9:
                coot['database'][key]['subset'] = 'training'
            else:
                coot['database'][key]['subset'] = 'validation'
        elif dataset_split == 2:
            if movie == val_movie:
                coot['database'][key]['subset'] = 'validation'
            else:
                coot['database'][key]['subset'] = 'training'


    # Join the scene_df with the shot_df
    shot_df = shot_df.merge(scene_df, on='scene_id', how='left')


    # Add the relative start_time and end_time to scene start
    shot_df['rel_start_time'] = shot_df['start_time'] - shot_df['scene_start_time']
    shot_df['rel_end_time'] = shot_df['end_time'] - shot_df['scene_start_time']

    ### GT DATAFRAME
    '''
    # Load book
    book_name = getBookName(movie)
    book_file = scipy.io.loadmat(book_name)
    num_sents = len(book_file['book']['sentences'].item()['sentence'])
    # Get annotations
    file_anno = anno_path + movie + '-anno.mat'
    anno_file = scipy.io.loadmat(file_anno)
    annotation = anno_file['anno']
    num_align = annotation.shape[1]
    '''
print(total_scene_duration)
