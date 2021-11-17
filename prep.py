import torch
import numpy as np
import torchvision.models.vgg as models
import torchvision.transforms as transforms
from PIL import Image
import json
import pickle
import scipy.io

dataset_path = 'flickr8k'

#Images
image_data = scipy.io.loadmat(dataset_path + '/vgg_feats.mat')
images = image_data['feats']

#Captions
with open(dataset_path + '/dataset.json') as f:
	caption_data = json.load(f)['images']

S = images.shape[1]
T = int(S * 0.9)
V = S - T

train_imgs = np.zeros((T, 4096))
train_caps = []
train_names = []

val_imgs = np.zeros((V, 4096))
val_caps = []
val_names = []

for image_id in range(images.shape[1]):
    if image_id < T:
        train_imgs[image_id] = images[:, image_id]
        train_names.append(caption_data[image_id]['filename'])
        for i in range(5):
            train_caps.append(caption_data[image_id]['sentences'][i]['raw'])
    else:
        val_imgs[image_id - T] = images[:, image_id]
        val_names.append(caption_data[image_id - T]['filename'])
        for i in range(5):
            val_caps.append(caption_data[image_id - T]['sentences'][i]['raw'])

print('Train Images shape:', train_imgs.shape, 'Train Length of captions:', len(train_caps))
print('Train Images shape:', val_imgs.shape, 'Train Length of captions:', len(val_caps))

# Saving Train
with open('data/' + dataset_path + '/train/image_features.pkl', 'wb') as f:
	pickle.dump(train_imgs, f)
with open('data/' + dataset_path + '/train/captions.pkl', 'wb') as f:
	pickle.dump(train_caps, f)
with open('data/' + dataset_path + '/train/file_names.pkl', 'wb') as f:
	pickle.dump(train_names, f)

# Saving Val
with open('data/' + dataset_path + '/val/image_features.pkl', 'wb') as f:
	pickle.dump(val_imgs, f)
with open('data/' + dataset_path + '/val/captions.pkl', 'wb') as f:
	pickle.dump(val_caps, f)
with open('data/' + dataset_path + '/val/file_names.pkl', 'wb') as f:
	pickle.dump(val_names, f)

print('Saved')
