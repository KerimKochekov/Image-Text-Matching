import torch
from torch import nn
import numpy as np
from torch import optim
import torchvision.models.vgg as models
import torchvision.transforms as transforms
from model import ImgEncoder, SentenceEncoder
from utils import get_hot, triplet_loss_cap, triplet_loss_img, build_dictionary, Score
from PIL import Image
import pickle
import json

# Parameters
margin = 0.05
max_epochs = 10
dim_image = 4096
batch_size = 128
dim = 1024
dim_word = 256
lrate = 0.001
dataset_path = 'flickr8k'

# Loading the dataset
print('Loading the train dataset')
with open('data/' + dataset_path + '/train/image_features.pkl', 'rb') as f:
	train_images = pickle.load(f)
	train_images = train_images.astype(np.float32)


with open('data/' + dataset_path + '/train/captions.pkl', 'rb') as f:
	train_caps = pickle.load(f)
	
print('Images shape:', train_images.shape, 'Length of captions:', len(train_caps))

print('Loading the val dataset')
with open('data/' + dataset_path + '/val/image_features.pkl', 'rb') as f:
	val_images = pickle.load(f)
	val_images = val_images.astype(np.float32)

with open('data/' + dataset_path + '/val/captions.pkl', 'rb') as f:
	val_caps = pickle.load(f)

print('Val Images shape:', val_images.shape, 'Length of Val captions:', len(val_caps))

# Creating dictionary and saving
print('Creating the word dictionary')
worddict = build_dictionary(train_caps)
with open('worddict.pkl', 'wb') as f:
	pickle.dump(worddict, f)
print('Dictionary size:', len(worddict))

# Loading models
ImgEncoder = ImgEncoder(dim_image, dim)
SentenceEncoder = SentenceEncoder(len(worddict)+2, dim_word, dim)
print('Models loaded')

# Adam Optimizer
optimizer = optim.Adam(list(ImgEncoder.parameters()) + list(SentenceEncoder.parameters()), lr = lrate)
print('Loaded Adam optimizer')

# Training
print('Training begins')
epochLoss = []
meanRank = []
for epoch in range(max_epochs):
	
	print('Epoch:', epoch)
	totalLoss = 0

	for batch_index in range(0, train_images.shape[0], batch_size):
		
		if batch_index + batch_size >= train_images.shape[0]:
			break

		# Data preproc
		curr_ims = train_images[batch_index:batch_index+batch_size]
		all5_caps = train_caps[5*batch_index:5*(batch_index+batch_size)]
		curr_caps = []
		for i in range(batch_size):
			curr_caps.append(all5_caps[5*i + np.random.randint(0, 5)])

		one_hot_caps = []
		for i in range(batch_size):
			one_hot_caps.append(get_hot(curr_caps[i], worddict))

		# Encoding
		encoded_ims = ImgEncoder(torch.from_numpy(curr_ims))
		encoded_caps = []
		for i in range(batch_size):
			encoded_caps.append(SentenceEncoder(one_hot_caps[i]))
		encoded_caps = torch.stack(encoded_caps).reshape(batch_size, dim)

		# Real training
		optimizer.zero_grad()

		# Calculating Loss
		loss = 0
		for i in range(batch_size):
			# Image as anchor
			anchor = encoded_ims[i:i+1].repeat(batch_size - 1, 1)
			positive = encoded_caps[i:i+1].repeat(batch_size - 1, 1)
			negative = torch.cat((encoded_caps[:i], encoded_caps[i+1:]), 0)
			loss += triplet_loss_img(anchor, positive, negative, margin)

			# Caption as anchor
			anchor = encoded_caps[i:i+1].repeat(batch_size - 1, 1)
			positive = encoded_ims[i:i+1].repeat(batch_size - 1, 1)
			negative = torch.cat((encoded_ims[:i], encoded_ims[i+1:]), 0)
			loss += triplet_loss_cap(anchor, positive, negative, margin)

		# Logging
		totalLoss += loss.item()
		print('Samples seen: ' + str(batch_index+batch_size) +  '/' + str(train_images.shape[0]), 'loss:', loss.item())

		# Updating weights
		loss.backward()
		optimizer.step()

	# Logging for early stopping
	print('Training loss:', totalLoss)
	epochLoss.append(totalLoss)

	# Ranks on test set
	r = []
	encoded_val_ims = ImgEncoder(val_images)
	for i in range(len(val_caps)):
		hot = get_hot(val_caps[i], worddict)
		encoded_val_cap = SentenceEncoder(hot).repeat(val_images.shape[0], 1)
		S = Score(encoded_val_cap, encoded_val_ims)
		ranks = S.argsort().cpu().numpy()[::-1]
		r.append(np.where(ranks==i//5)[0][0] + 1)
	
	print('Mean rank on val set: ' + str(np.mean(np.array(r))) + '/' + str(val_ims.shape[0]))
	meanRank.append(np.mean(np.array(r)))

# Saving models
print("Training Completed!")
print('Loss over epochs')
print(epochLoss)
print('Mean rank over epochs')
print(meanRank)
torch.save(ImgEncoder.state_dict(), 'ImgEncoder.pt')
torch.save(SentenceEncoder.state_dict(), 'SentenceEncoder.pt')