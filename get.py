import random
import numpy as np
from model import ImageEncoder, SentencesEncoder
import torch
import pickle
from utils import get_hot, Score

print('Loading the train dataset')
with open('vgg16/train/image_features.pkl', 'rb') as f:
	train_images = pickle.load(f)
	train_images = train_images.astype(np.float32)
	train_images = train_images / torch.norm(torch.from_numpy(train_images), dim=1, p=2).reshape(-1, 1)
with open('data/flickr8k/train/captions.pkl', 'rb') as f:
	train_caps = pickle.load(f)
with open('data/flickr8k/train/file_names.pkl', 'rb') as f:
	train_names = pickle.load(f)
print('Images shape:', train_images.shape, 'Length of captions:', len(train_caps))

print('Loading the val dataset')
with open('vgg16/val/image_features.pkl', 'rb') as f:
	val_images = pickle.load(f)
	val_images = val_images.astype(np.float32)
	val_images = val_images / torch.norm(torch.from_numpy(val_images), dim=1, p=2).reshape(-1, 1)
with open('data/flickr8k/val/captions.pkl', 'rb') as f:
	val_caps = pickle.load(f)
with open('data/flickr8k/val/file_names.pkl', 'rb') as f:
	val_names = pickle.load(f)
print('Val Images shape:', val_images.shape, 'Length of Val captions:', len(val_caps))

with open('data/flickr8k/worddict.pkl', 'rb') as f:
	worddict = pickle.load(f)
print('Loaded dictionary')
print('Dictionary size:', len(worddict))

#Mergin train and validation datasets
images = torch.cat((train_images, val_images), 0)

caps = np.concatenate([train_caps, val_caps])
names = np.concatenate([train_names, val_names])
# Parameters
dim_image = 4096
dim = 1024
dim_word = 300

# Loading trained models
ImgEncoder = ImageEncoder(dim_image, dim)
ImgEncoder.load_state_dict(torch.load('new_model/flickr8k/ImgEncoder/ImgEncoder-70.pt', map_location=torch.device('cpu')))
SentenceEncoder = SentencesEncoder(len(worddict)+2, dim_word, dim)
SentenceEncoder.load_state_dict(torch.load('new_model/flickr8k/SentenceEncoder/SentenceEncoder-70.pt', map_location=torch.device('cpu')))
print('Models loaded')


input = open("uploads/description.txt", "r")
caption = str(input.read())
input.close()
print(caption)

encoded_images = ImgEncoder(images)
hot = get_hot(caption, worddict)
encoded_captions = SentenceEncoder(hot).repeat(images.shape[0], 1)
S = Score(encoded_captions, encoded_images)
ranks = S.argsort().cpu().numpy()[::-1]

ir = 1
K = 20
output = open("uploads/matched-images.txt", "w")
for ix in ranks:
	ir += 1
	output.write('Images/'+names[ix]+'\n')
	if ir == K:
		break