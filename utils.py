import torch
import numpy as np
import string
from torch import nn


def tokenize(text):
	table = str.maketrans('', '', string.punctuation)
	# tokenize
	desc = text.split()
	# to lower
	desc = [word.lower() for word in desc]
	# remove punctuation
	desc = [word.translate(table) for word in desc]
	# remove words less in len
	desc = [word for word in desc if len(word) > 1]
	# remove numbers
	desc = [word for word in desc if word.isalpha()]
	return desc


def build_dictionary(text):
    """
    Build a dictionary (mapping of tokens to indices)
    text: list of sentences (pre-tokenized)
    """
    wordcount = {}
    for cc in text:
        words = tokenize(cc)
        for word in words:
            if word not in wordcount:
                wordcount[word] = 0
            wordcount[word] += 1
		# print(words)

    words = list(wordcount.keys())
    freqs = list(wordcount.values())
    sorted_idx = np.argsort(freqs)[::-1]

    worddict = {}
    for idx, sidx in enumerate(sorted_idx):
        worddict[words[sidx]] = idx+2  # 0: <eos>, 1: <unk>

    return worddict

def get_hot(cap, worddict):
	x = np.zeros((len(cap.split())+1, len(worddict)+2))

	r = 0
	for w in cap.split():
		if w in worddict:
			x[r, worddict[w]] = 1
		else:
			# Unknown word/character
			x[r, 1] = 1
		r += 1
	# EOS
	x[r, 0] = 1

	return torch.from_numpy(x).float()

def Score(caps, imgs):
	z = torch.zeros(caps.shape)
	return -torch.sum(torch.max(z, caps-imgs)**2, dim=1)

def triplet_loss_img(anchor, positive, negative, margin):
	ps = Score(positive, anchor)
	pn = Score(negative, anchor)
	z = torch.zeros(ps.shape)
	return torch.sum(torch.max(z, margin - ps + pn))

def triplet_loss_cap(anchor, positive, negative, margin):
	ps = Score(anchor, positive)
	pn = Score(anchor, negative)
	z = torch.zeros(ps.shape)
	return torch.sum(torch.max(z, margin - ps + pn))
