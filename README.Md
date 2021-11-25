# Image-Text Matching

## Description

Pytorch code for training an image-text matching model on the 3 datasets(trained for now only in "Flickr8k"). Based on the paper [Order-Embeddings of Images and Language](https://arxiv.org/pdf/1511.06361.pdf). The model trained with **Triplet loss** in Cross-Modal environment and **Adam**.optimizer. The idea of Cross-Modal retrieval comes on representing image and text in same embedding space, so we can compare two embedded vectors with simple Cosine similarity. My previous apporaches on D1.2 and D1.3 was on implementing merged architecture for image and text with Image captioning, but that did not succeed, so found new paper(as I mentioned above) and tried to implement the idea below.

## Details

* Preprocessing(done by prep.py): The first step is to read and normalize read features from all the images. These features will be used for all further processing.

* Training(done by train.py): Loads the image-encoder and the sentence-encoder and trains the model using Order Embedding loss. No negative mining is being done in this code. The hyperparameters have been set according to the Research Paper.

* Testing(done by eval.py): Finds the mean retrieval rank against the test dataset.

Check utils.py for utility function definitions and model.py for encoder architecture.

## Pretrained model for image feature:
I analyzed 2 different VGG models with different applied techniques for getting the 4096 sized feature vector:
* The 16-layer [VGG network](http://www.robots.ox.ac.uk/~vgg/research/very_deep/) from Simonyan and Zisserman, because the model is beautiful, powerful and available with [Caffe].
* The 22-layer VGG model in Keras with pretrained weights [vgg16_weights_tf_dim_ordering_tf_kernels.h5](https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5
)

## How to Run:

```bash
bar@foo$:~/python3 prep.py
bar@foo$:~/python3 train.py
bar@foo$:~/python3 eval.py
```

## Performance
Trained for now only with Flickr8k dataset for 10 epochs due to technical limitations. Here is loss value over the steps for 10 epochs. We can easily see that loss function converges and reaches best possible version after training more.

![](https://github.com/KerimKochekov/PMLDL-Project-Image-Text-Matching/blob/main/loss.png)

A mean rank of 400 out of 800 images for a given caption can be easily achieved by training this model on "Flickr8k" dataset (90% training, and 10% validation. Flickr8k has 8k images, thus 800 images in validation).

## Datasets
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
