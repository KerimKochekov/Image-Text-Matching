import numpy as np
import torch
from torch import nn
from utils import get_hot, Score
from model import ImageEncoder, SentencesEncoder
import pickle

# parameters
dim_image = 4096
dim = 1024
dim_word = 300
path = 'data/flickr8k/'

# Loading the dataset
print('Loading the val dataset')
with open(path+'val/image_features.pkl', 'rb') as f:
	val_images = pickle.load(f)
	val_images = val_images.astype(np.float32)
	val_images = val_images/torch.norm(torch.from_numpy(val_images), dim=1, p=2).reshape(-1, 1)
with open(path+'/val/captions.pkl', 'rb') as f:
	caps = pickle.load(f)

print('Images shape:', val_images.shape, 'Length of captions:', len(caps))

# Loading dictionary
with open(path+'worddict.pkl', 'rb') as f:
	worddict = pickle.load(f)
print('Loaded dictionary')
print(len(worddict))

# Loading trained models
ImgEncoder = ImageEncoder(dim_image, dim)
ImgEncoder.load_state_dict(torch.load('new_model/flickr8k/ImgEncoder/ImgEncoder-70.pt', map_location=torch.device('cpu')))
SentenceEncoder = SentencesEncoder(len(worddict)+2, dim_word, dim)
SentenceEncoder.load_state_dict(torch.load('new_model/flickr8k/SentenceEncoder/SentenceEncoder-70.pt', map_location=torch.device('cpu')))
print('Models loaded')

# Data preproc
r = []
encoded_ims = ImgEncoder(val_images)
for i in range(len(caps)):
	hot = get_hot(caps[i], worddict)
	encoded_cap = SentenceEncoder(hot).repeat(val_images.shape[0], 1)
	S = Score(encoded_cap, encoded_ims)
	ranks = S.argsort().cpu().numpy()[::-1]
	r.append(np.where(ranks==i//5)[0][0] + 1)
	#print('Rank:', str(r[-1])+'/'+str(ranks.shape[0]))
print(np.mean(np.array(r)))