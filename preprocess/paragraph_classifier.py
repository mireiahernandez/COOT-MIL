import pandas as pd
import logging
import json
import os
import math
import glob
import numpy as np
import ipdb
import scipy.io
from collections import defaultdict

os.environ["CUDA_VISIBLE_DEVICES"]="1,2,3"

from simpletransformers.classification import (
    MultiLabelClassificationModel, MultiLabelClassificationArgs
)

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

labels_list = ['dialog', 'story', 'descriptionOfPlace', 'descriptionOfAppearance', 'descriptionOfAction', 'descriptionOfObject', 'descriptionOfSound']


'''
    PATHS AND NAMES
'''

# Data path
dataset_path = '/data/vision/torralba/datasets/movies/data/'
bksnmvs_path = '/data/vision/torralba/frames/data_acquisition/booksmovies/data/booksandmovies/'
anno_path = '{}/antonio/annotation/'.format(bksnmvs_path)
text_annotation_path = '/data/vision/torralba/movies-books/booksandmovies/joanna/bksnmovies/data/gt_alignment/consecutive_text_labels_v2'

# Model path
model_path = '/data/vision/torralba/scratch/mireiahe/bksandmvs/paragraph_cls/'
split = 0  # 0 for random-split, 1 for book-split
output_dir = model_path + 'roberta3_{}-split'.format("random" if split == 0 else "book")

# Movies
movies = ['American.Psycho','Brokeback.Mountain','Fight.Club','Gone.Girl','Harry.Potter.and.the.Sorcerers.Stone','No.Country.for.Old.Men','One.Flew.Over.the.Cuckoo.Nest','Shawshank.Redemption','The.Firm','The.Green.Mile','The.Road']
movies_titles = [movie.replace('.', '_') for movie in movies]
imbds = ['tt0144084','tt0388795','tt0137523','tt2267998','tt0241527','tt0477348','tt0073486','tt0111161','tt0106918','tt0120689','tt0898367']
dataset_split = 1 # 1 for 90% data from all movies in train and 10% in val; 2 for n-1 movies in train 1 in val
val_movie = movies[0]

# Load best model from training
model = MultiLabelClassificationModel('roberta', '{}/checkpoint-500-epoch-10'.format(output_dir), num_labels=7)

# Get book name
def getBookName(filen):
    filen.split('/')[-1]
    book_path = '{}/books/'.format(bksnmvs_path)
    book_name = filen+'.(hl-2).mat'
    return book_path+book_name


def wrangle(sentence):
    front, back, middle = False, False, False
    if repr(sentence)[0] == '"': front = True
    if repr(sentence)[-1] == '"': back = True
    if front: sentence = '"' + sentence
    if back: sentence += '"'
    return sentence


labeled_sents = dict()




# For each movie
for movie in movies:
    book_name = getBookName(movie)
    book_file = scipy.io.loadmat(book_name)
    book_dialogs = book_file['book']['dialog'][0][0]
    dialog_sent_id = [sent_id for dialog in book_dialogs for sent_id in range(dialog[0], dialog[1]+1)]
    num_sents = book_file['book']['sentences'][0][0].shape[0]
    sents = [wrangle(book_file['book']['sentences'][0][0][i,0][0][0]) for i in range(num_sents)]
    pred, raw_outputs = model.predict(sents)
    labeled_sents[movie] = {j: [labels_list[i] for i in range(0,7) if p[i] == 1] for j, p in enumerate(pred)}

json.dump(labeled_sents, open('labeled_sents.json', 'w'))
