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
dialog_anno = json.load(open('../data/bksnmovies/dialog_anno.json', 'r'))
labeled_sents = json.load(open('../data/bksnmovies/labeled_sents.json', 'r'))
movies = ['American.Psycho','Brokeback.Mountain','Fight.Club','Gone.Girl','Harry.Potter.and.the.Sorcerers.Stone','No.Country.for.Old.Men','One.Flew.Over.the.Cuckoo.Nest','Shawshank.Redemption','The.Firm','The.Green.Mile','The.Road']
movies_titles = [movie.replace('.', '_') for movie in movies]
imbds = ['tt0144084','tt0388795','tt0137523','tt2267998','tt0241527','tt0477348','tt0073486','tt0111161','tt0106918','tt0120689','tt0898367']
dataset_split = 2 # 1 for 90% data from all movies in train and 10% in val; 2 for n-1 movies in train 1 in val
val_movie = movies[0]

sel_movie = 'Harry.Potter.and.the.Sorcerers.Stone'

lines = ["<html>",
        "<h2>{}</h2>".format(sel_movie),
        "<table>",
        "<tbody>"
]

book_name = getBookName(sel_movie)
book_file = scipy.io.loadmat(book_name)
        

for i, movie in enumerate(movies):
    if movie == sel_movie:
        title = movies_titles[i]
        imbd = imbds[i]

        labels = dialog_anno[movie]
        
        sentences = book_file['book']['sentences'].item()['sentence']
        num_sent = len(sentences)
        
        for i in range(num_sent):
            sentence = book_file['book']['sentences'].item()['sentence'][i][0][0]
            label = labels[str(i)]
            if 'visual' in label:
                lines.append(f"<tr><td style=color:blue> {sentence} </td></tr>")
            elif 'dialog' in label:
                lines.append(f"<tr><td style=color:red> {sentence} </td></tr>")
            else:
                lines.append(f"<tr><td style=color:black> {sentence} </td></tr>")
lines.append('</tbody></table></html>')

with open('{}_parcls.html'.format(sel_movie), 'w') as out:
    out.writelines(lines)

cmd = f"mv {sel_movie}_parcls.html ~/public_html/{sel_movie}_parcls.html"
#os.system(cmd)
