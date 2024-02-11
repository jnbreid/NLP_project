
This directory contains notebooks and a .py file containing the code we wrote.

To execute the code with the data add a directory 'asc' with the data from https://github.com/akkarimi/BERT-For-ABSA/tree/master/asc on the level of the notebooks (level of this readme).

Each notebook contains the code needed to train and test one specific model. This includes data loading, preprocessing, model definitions, training/evaluation loops, ... .
Only the notebook 'optimizing_DeBERTa' contains the code to run the optimizer optuna for DeBERTa for two hyperparameters.

Specific content of each notebook can be seen below.



## DeBERTa.ipynb
Notebook containing our **main/best** results that we achived with DeBERTa. 

## optimizing_DeBERTa.ipynb
Notebook containing code to optimize the learning rate and number of epochs during the training of DeBERTa.


## BERT.ipynb
Training and testing of a BERT model with the given datasets.


## log_regression.ipynb
Using logistic regression together with TFIDF to classify sentiments (SVM can also be used)


## RoBERTa.ipynb
Training and testing of a RoBERTa model with the given datasets.


## dataset_util.py
This file contains utility functions for working with the dataset (e.g. preprocessing, augmentation).


## datasets
This directory only contains the precomputed augmented dataset files.
