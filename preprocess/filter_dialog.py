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


def isVisualDialog(labels):
    is_dialog = False
    is_visual = False
    for label in labels:
        if 'dialog' in label: is_dialog = True
        if 'description' in label: is_visual = True
    if is_visual and is_dialog: return ['dialog', 'visual']
    else: return ['dialog']

def isVisual(labels):
    is_visual = False
    for label in labels:
        if 'description' in label: is_visual = True
    if is_visual: return ['visual']
    else: return ['non-visual']




'''
    PATHS AND NAMES
'''

dataset_path = '/data/vision/torralba/datasets/movies/data/'
bksnmvs_path = '/data/vision/torralba/frames/data_acquisition/booksmovies/data/booksandmovies/'
anno_path = '{}/antonio/annotation/'.format(bksnmvs_path)
text_annotation_path = '/data/vision/torralba/movies-books/booksandmovies/joanna/bksnmovies/data/gt_alignment/consecutive_text_labels_v2'
labeled_sents = json.load(open('../data/bksnmovies/labeled_sents.json', 'r'))
movies = ['American.Psycho','Brokeback.Mountain','Fight.Club','Gone.Girl','Harry.Potter.and.the.Sorcerers.Stone','No.Country.for.Old.Men','One.Flew.Over.the.Cuckoo.Nest','Shawshank.Redemption','The.Firm','The.Green.Mile','The.Road']
wrong_movies = [3, 5, 10]
movies_titles = [movie.replace('.', '_') for movie in movies]
imbds = ['tt0144084','tt0388795','tt0137523','tt2267998','tt0241527','tt0477348','tt0073486','tt0111161','tt0106918','tt0120689','tt0898367']
dataset_split = 1 # 1 for 90% data from all movies in train and 10% in val; 2 for n-1 movies in train 1 in val
val_movie = movies[0]

labeled_sents = json.load(open('../data/bksnmovies/labeled_sents.json', 'r'))



# Filter dialog, visual, non-visual

dialog_anno = {}
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
    
    dialog_anno[movie] = {}
    
    # Filter out dialogs
    sentences = book_file['book']['sentences'].item()['sentence']
    is_dialog = False

    for j, sentence in enumerate(sentences):
        count = sentence[0][0].count('"')
        labels = labeled_sents[movie][str(j)]
        if count%2 == 1: # if there is an odd number of "
            if is_dialog: is_dialog = False # if there was a dialog going on, it has been closed
            else: is_dialog = True # otherwise, a dialog has been opened
            dialog_anno[movie][j] = isVisualDialog(labels)
        elif count == 0: # if there is no "
            if is_dialog: dialog_anno[movie][j] = ['dialog'] # if there was a dialog going on, still is
            else: dialog_anno[movie][j] = isVisual(labels) # otherwise, no dialog
        else: # if there is an even number of "
            is_dialog = False # dialogs have been opened and closed
            dialog_anno[movie][j] = isVisualDialog(labels)


path = '/data/vision/torralba/scratch/mireiahe/HierarchicalMIL/data/bksnmovies/dialog_anno.json'
json.dump(dialog_anno, open(path, 'w'))
