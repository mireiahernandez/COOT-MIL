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

def getVisualSents(start = -1, end = -1, num = 50, labels=[]):
    num_sents = len(list(labels.keys())) # total number of sentences
    a, b = start, end # where to look for visual sentences
    counter = 0 # counter of visual sentences
    index = start # index where to start looking
    ids = [] # ids of the visual sentences
    if end == -1: # if there is no end, stop at num_sents
        b = num_sents
        index = start
    if start == -1: # if there is no start, commence at end
        a = 0
        index  = end-1
    while counter < num and index in range(a,b):
        if isVisual(labels[str(index)]):
            counter += 1
            if start == -1: ids = [index] + ids # if there is no start, append backwards
            else: ids.append(index)
        if start == -1: index -=1 # if there is no start, move backwards
        else: index +=1 # else move forwards
    return ids

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
labeled_sents = json.load(open('../data/bksnmovies/labeled_sents.json', 'r'))
movies = ['American.Psycho','Brokeback.Mountain','Fight.Club','Gone.Girl','Harry.Potter.and.the.Sorcerers.Stone','No.Country.for.Old.Men','One.Flew.Over.the.Cuckoo.Nest','Shawshank.Redemption','The.Firm','The.Green.Mile','The.Road']
movies_titles = [movie.replace('.', '_') for movie in movies]
imbds = ['tt0144084','tt0388795','tt0137523','tt2267998','tt0241527','tt0477348','tt0073486','tt0111161','tt0106918','tt0120689','tt0898367']
dataset_split = 1 # 1 for 90% data from all movies in train and 10% in val; 2 for n-1 movies in train 1 in val
val_movie = movies[0]
'''
    GROUND-TRUTH
'''

'''
{'duration': 122.56, 
'subset': 'validation', 
'recipe_type': '113', 
'annotations': [{'segment': [16, 25], 'id': 0, 
'sentence': 'melt butter in a pan'}, 
{'segment': [31, 34], 'id': 1, 'sentence': 'place the bread in the pan'}, {'segment': [37, 41], 'id': 2, 'sentence': 'flip the slices of bread over'}, {'segment': [43, 51], 'id': 3, 'sentence': 'spread mustard on the bread'}, {'segment': [51, 57], 'id': 4, 'sentence': 'place cheese on the bread'}, {'segment': [57, 60], 'id': 5, 'sentence': 'place the bread on top of the bread'}], 'video_url': 'https://www.youtube.com/watch?v=oDsUh1es_lo'}
'''


# Ground-truth annotations
coot = defaultdict(dict)
text_data = {}
vid_data = {}
pos_data = {}

for i, movie in enumerate(movies):
    print('Getting ground-truth for movie: ' + movie)

    title = movies_titles[i]
    imbd = imbds[i]

    labels = labeled_sents[movie]
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
    
    # Load book
    book_name = getBookName(movie)
    book_file = scipy.io.loadmat(book_name)
    
    # Get annotations
    file_anno = anno_path + movie + '-anno.mat'
    anno_file = scipy.io.loadmat(file_anno)
    annotation = anno_file['anno']
    num_align = annotation.shape[1]
    
    seen_scenes = []
    
    # Get alignment pairs
    for i in range(num_align):
        if movie != 'The.Road' or i != 127: # missing data
            # Get alignment info
            DVS = [annotation['D'][0][i].item(), annotation['V'][0][i].item(), annotation['S'][0][i].item()]
            time_sec = annotation['time'][0][i].item()
            id_sentence = annotation['sentence'][0][i].item()-1
            id_paragraph = annotation['paragraph'][0][i].item()-1
            id_line = annotation['line'][0][i].item()-1
            id_srt = annotation['srt'][0][i].item()-1
            id_shot = annotation['shot'][0][i].item()-1
            sentence = book_file['book']['sentences'].item()['sentence'][id_sentence][0]
            try:
                line = book_file['book']['lines'].item()['text'][0, id_line][0]
            except:
                line = ''
            subtitle = srt['srt']['content'][0][id_srt]
            fr = srt['shot']['fr'].item()[0][0]
            tstmp = srt['shot']['time'][0][0][id_shot]


            # Get scene of alignment
            id_scene = shot_df['scene_id'][id_shot]
            scene_key = movie + '_' + str(id_scene)
            seen_scenes.append(scene_key)
        
            gt_anno = {
            'segment':[shot_df['rel_start_time'][id_shot], shot_df['rel_end_time'][id_shot]],
            'real_segment': [tstmp[0], tstmp[1]],
            'id_shot': id_shot,
            'id_sentence':id_sentence,
            'type': DVS,
            'time':time_sec - shot_df['scene_start_time'][id_shot],
            'alig_num':i}
            coot['database'][scene_key]['annotations'].append(gt_anno)
    
    for scene_key in seen_scenes:
        id_scene = int(scene_key.split('_')[1])
        annos = coot['database'][scene_key]['annotations']
        first_sent, last_sent = annos[0]['id_sentence'], annos[-1]['id_sentence']
        first_shot, last_shot = annos[0]['id_shot'], annos[-1]['id_shot']
        start_scene_shot, end_scene_shot = scene_df['start_shot'][id_scene], scene_df['end_shot'][id_scene]
        num_sents_left = first_shot - start_scene_shot
        num_sents_right = end_scene_shot - last_shot
        id_sents = []
        id_sents.extend(getVisualSents(end=first_sent, num=num_sents_left, labels=labels))
        id_sents.extend(getVisualSents(start=first_sent, end=last_sent, labels=labels))
        id_sents.extend(getVisualSents(start=last_sent, num=num_sents_right, labels=labels))
        # Add a bag of positive candidates for gt alignment
        if len(id_sents) > 0:
            text_data[scene_key] = {id_sent:book_file['book']['sentences'].item()['sentence'][id_sent][0][0] for id_sent in id_sents}
            if len(text_data[scene_key].items()) == 0: ipdb.set_trace()
            vid_data[scene_key] = {id_shot:[shot_df['rel_start_time'][id_shot], shot_df['rel_end_time'][id_shot]] for id_shot in range(start_scene_shot, end_scene_shot+1)}
            pos_data[scene_key] = []
            window = 6
            for anno in annos:
                # id of original shot and sentence ground-truth annotation
                id_shot = anno['id_shot']
                id_sentence = anno['id_sentence']
                # find index in id_sents
                index_sentence = locateInVector(id_sentence, id_sents)
                # get positive sentence indexes
                positive_index_sentences = list(range(max(0, index_sentence-3), min(len(id_sents)-1, index_sentence+3)))
                # get positive sentence ids
                positive_sentences = [id_sents[index] for index in positive_index_sentences]
                if anno['type'][1] == 0: # if not visual, also add positive surrounding shots
                    positive_shots = list(range(max(start_scene_shot, id_shot - 3), min(end_scene_shot, id_shot + 3)))
                else: # if visual, only the given shot
                    positive_shots = [id_shot]
                pos_data[scene_key].append({'positive_shots':positive_shots, 'positive_sentences':positive_sentences})


coot2 = defaultdict(dict)
for k,v in coot['database'].items():
    if k in pos_data.keys(): coot2['database'][k] = v
json.dump(coot2, open('../annotations/bksnmovies/bknmovies_v0_split_{}_nonempty.json'.format(dataset_split), 'w'))


json.dump(text_data, open('../annotations/bksnmovies/text_data.json', 'w'))
json.dump(vid_data, open('../annotations/bksnmovies/vid_data.json', 'w'))
json.dump(pos_data, open('../annotations/bksnmovies/pos_data.json', 'w'))
