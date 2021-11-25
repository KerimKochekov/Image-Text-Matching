# Image-Text Matching

## Description

Pytorch code for training an image-text matching model on the 3 datasets(trained for now only in "Flickr8k"). Based on the paper [Order-Embeddings of Images and Language](https://arxiv.org/pdf/1511.06361.pdf). The model trained with **Triplet loss** in Cross-Modal environment and **Adam**.optimizer. The idea of Cross-Modal retrieval comes on representing image and text in same embedding space, so we can compare two embedded vectors with simple Cosine similarity. My previous apporaches on D1.2 and D1.3 was on implementing merged architecture for image and text with Image captioning, but that did not succeed, so found new paper(as I mentioned above) and tried to implement the idea below.

## Details

* Preprocessing(done by prep.py): The first step is to read and normalize read features from all the images. These features will be used for all further processing.

* Training(done by train.py): Loads the image-encoder and the sentence-encoder and trains the model using Order Embedding loss. No negative mining is being done in this code. The hyperparameters have been set according to the Research Paper.

* Testing(done by eval.py): Finds the mean retrieval rank against the test dataset.

Check utils.py for utility function definitions and model.py for encoder architecture.

## Pretrained model for image feature:
My fundamental hypothesis is if I can get a very precise prediction on image classification, this may imply that we extract most important features about this image, then I can use these features to match with image description. To do that I decided to use VGG pre-trained modesl. Analyzed 2 different VGG models with different applied techniques for getting the 4096 sized feature vector:
* The 16-layer [VGG network](http://www.robots.ox.ac.uk/~vgg/research/very_deep/) from Simonyan and Zisserman, because the model is beautiful, powerful and available with [Caffe].
* The 22-layer VGG model in Keras with pretrained weights [vgg16_weights_tf_dim_ordering_tf_kernels.h5](https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5
)

## Architecture
The encoder model for image and text uses basic Linear model with GRU, you can view the code of model in Python below:
```python
class ImgEncoder(nn.Module):
	def __init__(self, EMBEDDING_SIZE, COMMON_SIZE):
		super(ImgEncoder, self).__init__()
		self.linear = nn.Linear(EMBEDDING_SIZE, COMMON_SIZE)

	def forward(self, x):
		return self.linear(x).abs()

class SentenceEncoder(nn.Module):
	def __init__(self, VOCAB_SIZE, WORD_EMBEDDING_SIZE, COMMON_SIZE):
		super(SentenceEncoder, self).__init__()
		self.embed = nn.Linear(VOCAB_SIZE, WORD_EMBEDDING_SIZE)
		self.encoder = nn.GRU(WORD_EMBEDDING_SIZE, COMMON_SIZE)

	def forward(self, x):
		x = self.embed(x)
		o, h = self.encoder(x.reshape(x.shape[0], 1, x.shape[1]))
		return h.reshape(1, -1).abs()
```

## Achieving main goal with Triplet loss
My main goal for implementing this project to match images and texts somehow in same embedding space. Since they are different particles that can not be compared easily, so I wanted to apply somehow Cross-Modal retrieval idea to project both of the medias into same spaces. We can easily see that naive idea of turning image into vector space with CNN and turning text into another vectore space with RNN can not be compared. That is why I tried to develop a loss function that can compare bimodal architecture and penalizes each Encoder to encode image and text vector in same space.

## How to Run:

```bash
bar@foo$:~/python3 prep.py
bar@foo$:~/python3 train.py
bar@foo$:~/python3 eval.py
```

## Performance
In first phase trained model on Flickr8k dataset dataset for 10, 30 and 50 epochs respecitevely. The dataset has near 8k images, so it did not take my more than 2 hours to train for 50 epochs. Unlike in second phase trained model on Flickr30k dataset for 10 epochs, which itself took more than one hour of trainin, so I stopped there.

We can easily see for both datasets the loss function decreases by the time. Here is the loss function for Flickr30k for 10 epochs.

![](https://github.com/KerimKochekov/PMLDL-Project-Image-Text-Matching/blob/main/loss_flickr30k.png)

A mean rank for validation accuracy always smaller than 50%, which shows us image can find its belonging caption with probability at least 50%.

## Using the model to match on new images and captions
The code allows you to easily output the embedded vector representation of image and text on trained Flickr8k, Flick30K images and captions. If you want to run the code on arbitrary image and text (e.g. on your file system), things get a little more complicated because we need to first need to pipe your image through the VGG CNN to get the 4096 dimensional activations on top.

## You can find my all trained models here
[Flickr8k(50 epochs) and Flickr30k(10epochs)](https://disk.yandex.com.tr/d/lKpZNl3Zg0DflA)

## Datasets
Divided all datasets in form of 90% for training and 10% validation.
Used ready dataset prepared by Karpathy
* [Flickr8k](https://cs.stanford.edu/people/karpathy/deepimagesent/flickr8k.zip)
* [Flickr30k](https://cs.stanford.edu/people/karpathy/deepimagesent/flickr30k.zip)
* [COCO](https://cs.stanford.edu/people/karpathy/deepimagesent/coco.zip)

## Built Using

* [Python3](https://www.python.org)
* [PyTorch](https://pytorch.org/)
* [Numpy](https://numpy.org/)

## Team

[Kerim Kochekov](https://github.com/KerimKochekov)
