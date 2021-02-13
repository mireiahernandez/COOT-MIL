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


    ### GT DATAFRAME
    
    # Load book
    book_name = getBookName(movie)
    book_file = scipy.io.loadmat(book_name)
    
    # Get annotations
    file_anno = anno_path + movie + '-anno.mat'
    anno_file = scipy.io.loadmat(file_anno)
    annotation = anno_file['anno']
    num_align = annotation.shape[1]

    # Get alignment pairs
    for i in range(num_align):
        print(f"Alignment number {i}")
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

            for id_sent in range(id_sentence-5, id_sentence+5):
                if id_sent == id_sentence: print(30*'-')
                if id_sent == id_sentence + 1: print(30*'-')
                print(f"{id_sent}: {book_file['book']['sentences'].item()['sentence'][id_sent][0][0]}")
            correct_id_sent = int(input("Select the correct id_sent\n"))
            annotation['sentence'][0][i] = correct_id_sent + 1
        if i%20 == 0:
            scipy.io.savemat("annotation/{}-anno-{}.mat".format(movie, i), anno_file)
            anno_file = scipy.io.loadmat("annotation/{}-anno-{}.mat".format(movie, i))
