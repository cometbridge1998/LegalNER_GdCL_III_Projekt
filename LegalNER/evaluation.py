# Tools for evaluation, which can be imported elsewhere 

import pandas as pd
import numpy as np
import scipy as sp
from scipy import sparse
import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import Perceptron, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC



def get_all_labels(label_list):
    labels = []
    for i in range (len(label_list)):
        label = []
        if label_list[i].startswith("B"):
            label.append([i, label_list[i]])
            
            c = i + 1
            while True:
                if label_list[c].startswith("I"):
                    label.append([c, label_list[c]])
                else:
                    break
                c += 1
                
            labels.append(label)
            label = []
    return labels


def get_classify_report(true, pred, report=False):
    
    detected_true = []
    valid_pred = []

    for l in range(len(pred)):
        for t in range(len(true)):
            # Check whether they cover the exact same span or not:
            if [label[0] for label in pred[l]] == [label[0] for label in true[t]]:
                # The labels will be combined. 
                # detected_true and valid_pred are list of the names of entities [CASE_NUMBER, COURT ...]
                detected_true.append(true[t][0][1][2:])
                valid_pred.append(pred[l][0][1][2:])
                
    if len(detected_true) == len(valid_pred):
        if report == True:
            classes = sorted(list(set(valid_pred + detected_true)))
            print(classification_report(y_pred = valid_pred, y_true = detected_true, labels = classes))
        return detected_true, valid_pred
    else:
        raise ValueError


def get_recognition_report(true, pred, report=False):
    
    # create a dictionary to save the union of all entites in pred and in true. 
    # keys are the spans of each entites, values are dictionaries inside, 
    # which save the labels of these entities in the true list and pred list. 
    all_label_spans = {}
    
    for i in range(len(pred)):
        label = pred[i]
        label_span = [token[0] for token in label]
        all_label_spans[str(label_span)] = {"pred": pred[i][0][1][2:]}
        all_label_spans[str(label_span)]["true"] = "o"

    for i in range(len(true)):
        label = true[i]
        label_span = [token[0] for token in label]
        if str(label_span) in all_label_spans:
            all_label_spans[str(label_span)]["true"] = true[i][0][1][2:]
        else:
            all_label_spans[str(label_span)] = {"true": true[i][0][1][2:]}
            all_label_spans[str(label_span)]["pred"] = "o"
    
    all_labels_compare = pd.DataFrame.from_dict(all_label_spans).transpose()
    
    if report == True:
        classes = sorted(list(set(all_labels_compare["true"])))
        classes.remove("o")
        print(classification_report(y_pred = all_labels_compare["pred"], y_true = all_labels_compare["true"], labels = classes))
        
    return all_labels_compare


def get_confusion_matrix(pred, true, cat="natural_person"):
    
    juridical_person = ["COURT", "GPE", "ORG"]
    formats = ["CASE_NUMBER", "PRECEDENT", "PROVISION", "STATUTE", "DATE"]
    natural_person = ["JUDGE", "OTHER_PERSON", "PETITIONER", "RESPONDENT", "WITNESS"]
    labels = []
    display_labels = []
    
    if len(pred) == len(true):
        
        if cat == "natural_person":
            labels += natural_person
            display_labels += ["JUDGE", "OTHER", "PETITIONER", "RESPONDENT", "WITNESS"]
        elif cat == "juridical_person":
            labels += juridical_person
            display_labels += labels
        elif cat == "formats":
            labels += formats
            display_labels += ["CASE", "PRECEDENT", "PROVISION", "STATUTE", "DATE"]
        else:
            labels = natural_person + juridical_person + formats
            display_labels += labels
        
        cm = confusion_matrix(y_pred = pred, y_true = true, labels = labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = display_labels)
        disp.plot()
        
    else:
        raise ValueError





