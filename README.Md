# ImageCaptionRetrieval

## Description

Pytorch code for training an image-caption retrieval model on the MS-COCO dataset. Based on the paper [ORDER-EMBEDDINGS OF IMAGES AND LANGUAGE](https://arxiv.org/pdf/1511.06361.pdf). The model can be trained with 2 approaches:

* Symmetric Distance(Cosine loss)
* Ordered Distance

Furthermore, the code is also capable of fine-tuning the model using the VIST dataset.

## Details

* Preprocessing(done by prep.py): The first step is to extract out normalized 10-CROP VGG19 features from all the images. These features will be used for all further processing.

* Training(done by train.py): Loads the image-encoder and the sentence-encoder and trains using triplet loss alongwith Cosine Loss or orderEmbedding loss. No negative mining is being done in this code. The hyperparameters have been set according to the Research Paper.

* Testing(done by eval.py): Finds the mean retrieval rank against the test dataset.

Check utils.py for utility function definitions and model.py for encoder architecture.

## How to Run:

```bash
bar@foo$:~/python3 prep.py
bar@foo$:~/python3 train.py
bar@foo$:~/python3 eval.py
```

## Performance

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

## Author

[Kerim Kochekov](https://github.com/KerimKochekov)
