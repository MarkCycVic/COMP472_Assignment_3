"""
Assignment 3, COMP 472
Due December 13th, 2020
Use of Python 3.8
@author: Marc Vicuna, 40079109, El Hassan Ait Ouaziz, 26791573
"""

import numpy as np
import pandas as pd

class Model:
    # training sub-function
    # add one document to update vocubulary
    def add_document(doc, lab, ov):
        doc = doc.casefold()
        for word in doc.split(' '):
            if word in ov:
                ov[word] += lab
            else:
                ov[word] = lab
    # making the vocubulary through training. Documents and labels of dataset, d is the smoothing
    def make_vocabulary(documents, labels, d = 0.01):
        ov = {}
        priors = np.zeros(2)
        for doc, lab in zip(documents, labels):
            if lab == 'yes':
                priors[0] += 1
                Model.add_document(doc, np.array([1,0], dtype = 'i4'), ov)
            else:
                priors[1] += 1
                Model.add_document(doc, np.array([0,1], dtype = 'i4'), ov)
        priors = priors/np.sum(priors)
        #Construct fv vocabulary, find word count for each category, for each vocabulary.
        fv = {}
        ov_sum = np.zeros(2, dtype = 'i4')
        fv_sum = np.zeros(2, dtype = 'i4')
        for word, freq in ov.items():
            ov_sum += freq
            if np.sum(freq) > 1:
                fv[word] = freq
                fv_sum += freq
        # vocabulary size
        V_ov = len(ov)
        V_fv = len(fv)
        # conditional probability on ov
        for word, freq in ov.items():
            ov[word] = np.log10(np.array([(freq[0]+d)/(ov_sum[0]+d*V_ov),(freq[1]+d)/(ov_sum[1]+d*V_ov)]))
        # conditional probability on fv
        for word, freq in fv.items():
            fv[word] = np.log10(np.array([(freq[0]+d)/(fv_sum[0]+d*V_fv),(freq[1]+d)/(fv_sum[1]+d*V_fv)]))
        return ov, fv, priors
    #testing subfunction, evaluates a single document
    def evaluate_document(doc, v, priors):
        prob = np.log10(priors.copy())
        doc = doc.casefold()
        for word in doc.split(' '):
            if word in v:
                prob += v[word]
        if np.argmax(prob) == 0:
            return 'yes', prob[0]
        else:
            return 'no', prob[1]
    # Test model
    # Generate Trace files, tests the algorithm and prints results in the file.
    def trace(filename, IDs, documents, labels, v, priors):
        
        f = open(filename, 'w')
        for ID, doc, true_lab in zip(IDs, documents, labels):
            estim_lab, prob = Model.evaluate_document(doc, v, priors)
            if estim_lab == true_lab:
                match = 'correct'
            else:
                match = 'wrong'
            f.write('{}  {}  {:.2E}  {}  {}\n'.format(ID, estim_lab, 10**prob, true_lab, match))

    # Generate Evaluation files
    def evalation(input_file, output_file):
        data = pd.read_csv(input_file, sep='  ', header = None, usecols=[1,3,4], engine='python')
        estim_l = data[1].values
        true_l = data[3].values
        match = data[4].values
        accuracy, precision, recall = 0.0, np.array([0,0], dtype = 'f4'), np.array([0,0], dtype = 'f4')
        for est, tru, mat in zip(estim_l, true_l, match):
            if mat == 'correct':
                accuracy += 1
                if est == 'yes':
                    precision[0] += 1
                else:
                    precision[1] += 1
                if tru == 'yes':
                    recall[0] += 1
                else:
                    recall[1] += 1
        print(precision)
        print(recall)
        accuracy /= len(match)
        precision /= len(match)
        recall /= len(match)
        F1 = [2/(precision[0]**-1+recall[0]**-1),2/(precision[1]**-1+recall[1]**-1)]
        f = open(output_file, 'w')
        f.write('{:.4}\n'.format(accuracy))
        for metric in [precision, recall, F1]:
            f.write('{:.4}  {:.4}\n'.format(metric[0], metric[1]))