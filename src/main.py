"""
Assignment 3, COMP 472
Due December 13th, 2020
Use of Python 3.8
@author: Marc Vicuna, 40079109, El Hassan Ait Ouaziz, 26791573
"""
#imports
from Data import Data
from Model import Model
#main

# process data
train_file = 'data/covid_training.tsv'
train_documents, train_labels = Data.read_train_data(train_file)

test_file = 'data/test_set01.tsv'
test_IDs, test_documents, test_labels = Data.read_test_data(test_file)

# make model
ov, fv, priors = Model.make_vocabulary(train_documents, train_labels)

# test model
Model.trace('trace/trace_NB-BOW-OV.txt', test_IDs, test_documents, test_labels, ov, priors)
Model.trace('trace/trace_NB-BOW-FV.txt', test_IDs, test_documents, test_labels, fv, priors)

# evaluate the model
Model.evalation('trace/trace_NB-BOW-OV.txt', 'evaluation/eval_NB-BOW-OV.txt')
Model.evalation('trace/trace_NB-BOW-FV.txt', 'evaluation/eval_NB-BOW-FV.txt')