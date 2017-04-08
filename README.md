# AS-Reader in tensorflow and keras

## About This Repo

This is the tensorflow/keras version implementation/reproduce of the Attention Sum Reader model as presented in "Text Comprehension with the Attention Sum Reader Network" available at [http://arxiv.org/abs/1603.01547](http://arxiv.org/abs/1603.01547). 

And the original implementation is in [https://github.com/rkadlec/asreader](https://github.com/rkadlec/asreader).

## Quick Start

#### 1.Getting data

- Download CBT dataset, other dataset is not implemented.[https://github.com/rkadlec/asreader/blob/master/data/prepare-cbt-data.sh](https://github.com/rkadlec/asreader/blob/master/data/prepare-cbt-data.sh)

#### 2.Install dependencies

- Ensure you have installed python3.5 in your computer.
- Install keras and theano and other necessary packages.

`pip install -r requirements.txt`

- Install nltk punkt for tokenizer.

`python -m nltk.downloader punkt`

#### 3.Train the model

You can now train the model by entering the following command.

`python main.py --data_dir data_path --train_file **.txt --valid_file **.txt`

## Details

For the algorithm details, see the paper. And for other detail in the code, see the README of the original implementation.



