
## TODO
- test if notebooks can all be executed correctly


-------------------------------------------------------------------------------------
This directory contains notebooks containing the code we wrote.

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


## datasets
This directory only contains the dataset and is only included for convenient execution of the notebooks as the files are small.
This directory does **not** contain any of our work.
