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

lines = ["<html>",
        "<h2>Correct alignments</h2>",
        "<table>",
        "<tbody>"
]

sel_movie = 'Harry.Potter.and.the.Sorcerers.Stone'

for j, movie in enumerate(movies):
    if movie == sel_movie:
        print('Getting ground-truth for movie: ' + movie)

        title = movies_titles[j]
        imbd = imbds[j]

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

        # Get video
        title = movies_titles[j]
        # Get mp4
        print('{}/movies/{}/*.mp4'.format(dataset_path, title))
        vid_movie = glob.glob('{}/movies/{}/*.mp4'.format(dataset_path, title))
        if vid_movie == []: vid_movie = glob.glob('{}/movies/{}/*.m4v'.format(dataset_path, title))
        if vid_movie == []: vid_movie = glob.glob('{}/movies/{}/*.avi'.format(dataset_path, title))
        vid_movie = vid_movie[0]
        print('vid_movie is: {}'.format(vid_movie))

        # Get alignment pairs
        for i in range(num_align):
            if i in range(0, 150):
            #if movie != 'The.Road' or i != 127: # missing data

                DVS = [annotation['D'][0][i].item(), annotation['V'][0][i].item(), annotation['S'][0][i].item()]
                time_sec = annotation['time'][0][i].item()
                id_sentence = annotation['sentence'][0][i].item()-1
                id_paragraph = annotation['paragraph'][0][i].item()-1
                id_line = annotation['line'][0][i].item()-1
                id_srt = annotation['srt'][0][i].item()-1
                id_shot = annotation['shot'][0][i].item()-1
                sentence = book_file['book']['sentences'].item()['sentence'][id_sentence][0]
                prev2_sentence =  book_file['book']['sentences'].item()['sentence'][id_sentence-2][0]
                prev_sentence = book_file['book']['sentences'].item()['sentence'][id_sentence-1][0]
                next_sentence = book_file['book']['sentences'].item()['sentence'][id_sentence+1][0]
                next2_sentence = book_file['book']['sentences'].item()['sentence'][id_sentence+2][0]
                try:
                    line = book_file['book']['lines'].item()['text'][0, id_line][0]
                except:
                    line = ''
                subtitle = srt['srt']['content'][0][id_srt]
                fr = srt['shot']['fr'].item()[0][0]
                [clip_start, clip_end] = srt['shot']['time'][0][0][id_shot]
                clip_dur = clip_end-clip_start
                [clip_start_p, clip_end_p] = srt['shot']['time'][0][0][id_shot-1]
                clip_dur_p = clip_end_p-clip_start_p
                [clip_start_a, clip_end_a] = srt['shot']['time'][0][0][id_shot+1]
                clip_dur_a = clip_end_a-clip_start_a
                [clip_start_pp, clip_end_pp] = srt['shot']['time'][0][0][id_shot-2]
                clip_dur_pp = clip_end_pp-clip_start_pp
                [clip_start_aa, clip_end_aa] = srt['shot']['time'][0][0][id_shot+2]
                clip_dur_aa = clip_end_aa-clip_start_aa
                if DVS[1] == 1:
                    out_name = '~/public_html/tmp2/{}_{}_0.mp4'.format(movie, i)
                    out_name_p = '~/public_html/tmp2/{}_{}_01.mp4'.format(movie, i)
                    out_name_a = '~/public_html/tmp2/{}_{}_1.mp4'.format(movie, i)
                    
                    out_name_pp = '~/public_html/tmp2/{}_{}_02.mp4'.format(movie, i)
                    out_name_aa = '~/public_html/tmp2/{}_{}_2.mp4'.format(movie, i)

                    #print(out_name)
                    cmd =  'ffmpeg -n -nostats -loglevel 0 -ss {} -i {} -t {} -vcodec libx264 -acodec aac -strict -2 {}'.format(clip_start, vid_movie, clip_dur, out_name)
                    cmdp =  'ffmpeg -n -nostats -loglevel 0 -ss {} -i {} -t {} -vcodec libx264 -acodec aac -strict -2 {}'.format(clip_start_p, vid_movie, clip_dur_p, out_name_p)
                    cmdpp =  'ffmpeg -n -nostats -loglevel 0 -ss {} -i {} -t {} -vcodec libx264 -acodec aac -strict -2 {}'.format(clip_start_pp, vid_movie, clip_dur_pp, out_name_pp)
                    cmda =  'ffmpeg -n -nostats -loglevel 0 -ss {} -i {} -t {} -vcodec libx264 -acodec aac -strict -2 {}'.format(clip_start_a, vid_movie, clip_dur_a, out_name_a)
                    cmdaa =  'ffmpeg -n -nostats -loglevel 0 -ss {} -i {} -t {} -vcodec libx264 -acodec aac -strict -2 {}'.format(clip_start_aa, vid_movie, clip_dur_aa, out_name_aa)

                    os.system(cmd)
                    os.system(cmdp)
                    os.system(cmdpp)
                    os.system(cmda)
                    os.system(cmdaa)
                    

                    
                    src_name = 'tmp2/{}_{}.mp4'.format(movie, i)
                    src_namep = 'tmp2/{}_{}_01.mp4'.format(movie, i)
                    src_namepp = 'tmp2/{}_{}_02.mp4'.format(movie, i)
                    src_namea = 'tmp2/{}_{}_1.mp4'.format(movie, i)
                    src_nameaa = 'tmp2/{}_{}_2.mp4'.format(movie, i)
                    
                    lines.append('<tr>')
                    lines.append('<table><theader> Alignment number {} ({}) </theader>'.format(i, DVS))
                    lines.append('<tbody>')
                    lines.append('<tr>')
                    lines.append('<td><div> <video preload="none" width="500px"; controls><source src="{}"></video></div><div>{}:{}</div></td>'.format(src_name, convertTime(clip_start),convertTime(clip_end)))
                    lines.append('<td><div> <video preload="none" width="500px"; controls><source src="{}"></video></div><div>{}:{}</div></td>'.format(src_namep, convertTime(clip_start),convertTime(clip_end)))
                    lines.append('<td><div> <video preload="none" width="500px"; controls><source src="{}"></video></div><div>{}:{}</div></td>'.format(src_namepp, convertTime(clip_start),convertTime(clip_end)))
                    lines.append('<td><div> <video preload="none" width="500px"; controls><source src="{}"></video></div><div>{}:{}</div></td>'.format(src_namea, convertTime(clip_start),convertTime(clip_end)))
                    lines.append('<td><div> <video preload="none" width="500px"; controls><source src="{}"></video></div><div>{}:{}</div></td>'.format(src_nameaa, convertTime(clip_start),convertTime(clip_end)))

                    lines.append('</tr><tr>')
                    lines.append('<div style=color:black>Prev2 sentence: {}</div>'.format(prev2_sentence))

                    lines.append('<div style=color:black>Prev sentence: {}</div>'.format(prev_sentence))
                    lines.append('<div style=color:black>Sentence: {}</div>'.format(sentence))
                    lines.append('<div style=color:black>Next sentence: {}</div>'.format(next_sentence))
                    lines.append('<div style=color:black>Next2 sentence: {}</div>'.format(next2_sentence))


                    lines.append('<div style=color:black>Line {}: {}</div>'.format(id_line, line))
                    lines.append('</tr>')
                    lines.append('</tbody></table></tr>')

lines.append('</tr>')
lines.extend(['</tbody>', '</table>'])
ll = []
for i in lines:
    ll.append(i+ '\n')

with open('{}_corr.html'.format(sel_movie), 'w') as out:
    out.writelines(ll)
    
os.system('mv {}_corr.html ~/public_html/{}_corr.html'.format(sel_movie, sel_movie))
