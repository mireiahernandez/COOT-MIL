import numpy as np
import ipdb
dataset_path = '/data/vision/torralba/datasets/HowTo100m/'
features_path = dataset_path + 'howto100m_s3d_features/'

id_vid = 'zZzzp24pqlU.mp4.npy'

feats = np.load(features_path + id_vid)
ipdb.set_trace()
