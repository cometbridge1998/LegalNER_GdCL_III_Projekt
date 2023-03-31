#!/usr/bin/env python
# coding: utf-8

# # Model Selection

# compare the machine learning results of a <b>support vector machine</b> (SVC) of sklearn and the <b>sklearn_crfsuite</b> model, which is especially designed to learn the sequence of labels. <br>

# In[1]:


import nltk
import pandas as pd
import numpy as np
import scipy as sp
import sklearn
from scipy import sparse
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV


# In[2]:


from evaluation import get_all_labels, get_classify_report, get_recognition_report, get_confusion_matrix


# In[3]:


sparse_train = sp.sparse.load_npz('transitional_data/X5_train.npz')
sparse_dev = sp.sparse.load_npz('transitional_data/X5_dev.npz')


# In[4]:


y_train = pd.read_csv("transitional_data/y_train.csv")
y_dev = pd.read_csv("transitional_data/y_dev.csv")


# In[5]:


y_train = y_train.groupby(by = 'SentenceNR', group_keys=True).apply(lambda x: x)
y_train = y_train.rename(columns={"SentenceNR": "Sent", "Unnamed: 1": "TokenNr"})


# In[6]:


y_dev = y_dev.groupby(by = 'SentenceNR', group_keys=True).apply(lambda x: x)
y_dev = y_dev.rename(columns={"SentenceNR": "Sent", "Unnamed: 1": "TokenNr"})


# In[7]:


train_true = y_train["Label"]
dev_true = y_dev["Label"]


# In[ ]:





# ## Parameter fine tuning for the LinearSVC model

# In[8]:


param_grid = {
    "penalty": ["l1", "l2"],
    "loss": ["hinge", "squared_hinge"],
    "class_weight": ["balanced", None],
}


# In[24]:


svc = LinearSVC()


# In[7]:


if False:
    get_ipython().run_line_magic('%time', '')
    svc_gs = GridSearchCV(svc, param_grid, scoring = "f1_weighted", n_jobs = -1, refit='accuracy', cv=0, verbose=1 )
    svc_gs.fit(sparse_train, train_true)
    print(svc_gs.best_score_)
    print(svc_gs.best_params_)
    print(svc_gs.cv_results_)


# In[29]:


regularization_grid1 = {
    "penalty": ["l2"],
    "loss": ["hinge"],
    "class_weight": [None],
    "C": [0.001, 0.01, 0.1, 1],
}


# In[8]:


if False:
    get_ipython().run_line_magic('%time', '')
    svc_gs = GridSearchCV(svc, regularization_grid1, scoring = "f1_weighted", n_jobs = -1, refit='accuracy', cv=5, verbose=1 )
    svc_gs.fit(sparse_train, train_true)
    print(svc_gs.best_score_)
    print(svc_gs.best_params_)
    print(svc_gs.cv_results_)


# In[10]:


regularization_grid2 = {
    "penalty": ["l2"],
    "loss": ["hinge"],
    "class_weight": [None],
    "C": [0.2, 0.4, 0.6, 0.8, 1.0],
}


# In[11]:


if False:
    get_ipython().run_line_magic('%time', '')
    svc_gs = GridSearchCV(svc, regularization_grid2, scoring = "f1_weighted", n_jobs = -1, refit='accuracy', cv=5, verbose=1 )
    svc_gs.fit(sparse_train, train_true)
    print(svc_gs.best_score_)
    print(svc_gs.best_params_)
    print(svc_gs.cv_results_)


# In[12]:


regularization_grid3 = {
    "penalty": ["l2"],
    "loss": ["hinge"],
    "class_weight": [None],
    "C": [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55],
}


# In[13]:


if False:
    get_ipython().run_line_magic('%time', '')
    svc_gs = GridSearchCV(svc, regularization_grid3, scoring = "f1_weighted", n_jobs = -1, refit='accuracy', cv=5, verbose=1 )
    svc_gs.fit(sparse_train, train_true)
    print(svc_gs.best_score_)
    print(svc_gs.best_params_)
    print(svc_gs.cv_results_)


# ## best paramter combination according to the GridSearchCV
# #### {'C': 0.35, 'class_weight': None, 'loss': 'hinge', 'penalty': 'l2'}
# weighed average f1 score of classifying: 93% <br>
# weighed average f1 score of recognition: 67% <br>

# In[9]:


svc = LinearSVC(C=0.35, class_weight=None, loss="hinge", penalty="l2")
get_ipython().run_line_magic('time', 'svc.fit(sparse_train, train_true)')


# In[10]:


get_ipython().run_cell_magic('time', '', 'dev_pred = svc.predict(sparse_dev)\n')


# In[11]:


get_ipython().run_line_magic('time', 'labels_pred = get_all_labels(dev_pred)')
get_ipython().run_line_magic('time', 'labels_true = get_all_labels(dev_true.tolist())')


# In[12]:


detected_true, valid_pred = get_classify_report(labels_true, labels_pred, report=True)


# In[13]:


all_labels_compare = get_recognition_report(true = labels_true, pred = labels_pred, report=True)


# In[14]:


get_confusion_matrix(pred = all_labels_compare["pred"], true = all_labels_compare["true"], cat="natural_person")


# In[15]:


diff = all_labels_compare[all_labels_compare["pred"]!=all_labels_compare["true"]]
diff[diff["true"]=="PRECEDENT"]["pred"].value_counts()


# In[16]:


diffindex = list(diff[diff["true"]=="PRECEDENT"].index)


# In[20]:


df_dev = pd.read_csv("transitional_data/tokenized_dev.csv")
df_dev["Pred"] = dev_pred


# In[83]:


i = 17
begin = int(diffindex[i][1:-1].split(", ")[0]) - 5
end = int(diffindex[i][1:-1].split(", ")[-1]) + 8
df_dev[begin:end]


# 1. other person (respondent, petitioner, witness, judge) vs. precedent
# 0, 1(beispiel), 2, 3, 5, 7, 8, 9, 15, 25
# 
# 2. ORG, COURT, STATUTE, CASE_NUMBER, PROVISION vs. precedent
# 4, 5, 6, 7, 9, 10, 16, 22, 28, 41(beispiel)
# 
# 3. length -1 
# 11, 12, 13, 17(beispiel), 21, 23, 24, 51
# 
# 4. break
# 18, 19
# 
# 5. begin not found
# 23, 27, 28, 29(beispiel)
# 
# 6. connet
# 33, 34

# In[42]:





# In[ ]:





# In[ ]:





# ## Problem: Feature Selection
# Which features in the feature matrix make bigger contributions to the machine learning?

# In[39]:


from sklearn.feature_selection import SelectFromModel
X, y = sparse_train, train_true
print(f"X.shape: {X.shape}")
lsvc_l1 = LinearSVC( penalty='l1', loss = "squared_hinge", dual = False, C = 0.35).fit(X, y)
lsvc_l2 = LinearSVC( penalty='l2', loss = "hinge", C = 0.35).fit(X, y)


# In[42]:


# L1 norm
model_l1 = SelectFromModel(lsvc_l1, prefit=True)
X_train_new = model_l1.transform(X)
X_dev_new = model_l1.transform(sparse_dev)
print(f"X_train_new.shape: {X_train_new.shape}")
print(f"X_dev_new.shape: {X_dev_new.shape}")


# In[43]:


# L2 norm
model_l2 = SelectFromModel(lsvc_l2, prefit=True)
X_train_new = model_l2.transform(X)
X_dev_new = model_l2.transform(sparse_dev)
print(f"X_train_new.shape: {X_train_new.shape}")
print(f"X_dev_new.shape: {X_dev_new.shape}")


# In[ ]:





# In[28]:


from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
X_train, y_train = sparse_train, train_true
print(f"X_train.shape: {X_train.shape}")
clf = ExtraTreesClassifier(n_estimators=50)
clf = clf.fit(X, y)


# In[30]:


clf.feature_importances_.shape


# In[31]:


model = SelectFromModel(clf, prefit=True)
X_train_new = model.transform(X_train)
X_dev_new = model.transform(sparse_dev)
print(f"X_train_new.shape: {X_train_new.shape}")
print(f"X_dev_new.shape: {X_dev_new.shape}")


# In[32]:


get_ipython().run_cell_magic('time', '', 'dev_pred = clf.predict(sparse_dev)\n')


# In[33]:


get_ipython().run_line_magic('time', 'labels_pred = get_all_labels(dev_pred)')
get_ipython().run_line_magic('time', 'labels_true = get_all_labels(dev_true.tolist())')


# In[34]:


all_labels_compare = get_recognition_report(true = labels_true, pred = labels_pred, report=True)


# In[ ]:





# ## sklearn_crfsuite Model
# We will use the default feature matrix of the model. Details see below. <br>
# <i>NB: This Model is only compatible only with a sklearn version lower as 0.24</i> <br>

# weighed average f1 score of classifying: 95%  ( + 2% comparing to the svm model ) <br>
# weighed average f1 score of recognition: 75% ( + 8% comparing to the svm model) 

# In[8]:


sklearn.__version__


# In[7]:


import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics


# In[8]:


def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]


# <br>
# convert the dataframe to a structure required by the sklearn_crfsuite model.

# In[9]:


df_train = pd.read_csv("transitional_data/tagged_train_filled.csv", keep_default_na=False)
df_dev = pd.read_csv("transitional_data/tagged_dev_filled.csv", keep_default_na=False)


# In[10]:


train = df_train[["Token", "standard_tagger", "Label"]]
dev = df_dev[["Token", "standard_tagger", "Label"]]


# In[11]:


train = train.values.tolist()
dev = dev.values.tolist()


# In[12]:


def enumerate_tokens(sentence):
    c = 1
    for index, row in sentence.iterrows():
        sentence.at[index, 'TokenNR'] = c
        c += 1
    return sentence


# In[13]:


get_ipython().run_line_magic('time', "df_dev = df_dev.groupby(by = 'SentenceNR', group_keys=True).apply(enumerate_tokens)")
get_ipython().run_line_magic('time', "df_train = df_train.groupby(by = 'SentenceNR', group_keys=True).apply(enumerate_tokens)")


# In[14]:


train_SentenceNR = df_train.SentenceNR.unique()
dev_SentenceNR = df_dev.SentenceNR.unique()


# In[15]:


train_sents = [df_train[ df_train["SentenceNR"]==nr][["Token", "standard_tagger", "Label"]].values.tolist() for nr in train_SentenceNR ]
dev_sents = [df_dev[ df_dev["SentenceNR"]==nr][["Token", "standard_tagger", "Label"]].values.tolist() for nr in dev_SentenceNR ]


# In[16]:


get_ipython().run_cell_magic('time', '', 'X_train = [sent2features(s) for s in train_sents]\ny_train = [sent2labels(s) for s in train_sents]\n\nX_dev = [sent2features(s) for s in dev_sents]\ny_dev = [sent2labels(s) for s in dev_sents]\n')


# In[17]:


get_ipython().run_cell_magic('time', '', "crf = sklearn_crfsuite.CRF(\n    algorithm='lbfgs',\n    #{'c1': 0.17311222730699913, 'c2': 0.2034091783796535} according to the Hyperparameter Finetuning\n    c1=0.17311222730699913,\n    c2=0.2034091783796535,\n    max_iterations=100,\n    all_possible_transitions=True\n)\ncrf.fit(X_train, y_train)\n")


# In[22]:


y_pred = crf.predict(X_dev)


# <br>
# convert the label lists(y_dev(true), y_pred) back into a flat structure required by the evaluation tools.

# In[23]:


from functools import reduce
y_dev = reduce(lambda a,b:a+b, y_dev)
y_pred = reduce(lambda a,b:a+b, y_pred)


# In[26]:


y_pred = get_all_labels(y_pred)
y_true = get_all_labels(y_dev)


# In[27]:


detected_true, valid_pred = get_classify_report(y_true, y_pred, report=True)


# In[28]:


all_labels_compare = get_recognition_report(y_true, y_pred, report=True)


# In[33]:


get_confusion_matrix(pred = all_labels_compare["pred"], true = all_labels_compare["true"], cat="natural_person")


# In[ ]:





# ## Analysis
# crfsuirte provides two functions to analyse the quality of mashine learning.<br>
# <b>print_transitions</b> shows which labels are often mixed with each other. <br>
# <b>print_state_features</b> shows "typical" features of each label.

# In[34]:


from collections import Counter

def print_transitions(trans_features):
    for (label_from, label_to), weight in trans_features:
        print("%-6s \t->\t %-7s %0.6f" % (label_from, label_to, weight))

print("Top likely transitions:")
print_transitions(Counter(crf.transition_features_).most_common(10))


# In[35]:


def print_state_features(state_features):
    for (attr, label), weight in state_features:
        print("%0.6f %-8s %s" % (weight, label, attr))

print("Top positive:")
print_state_features(Counter(crf.state_features_).most_common(10))

print("\nTop negative:")
print_state_features(Counter(crf.state_features_).most_common()[-10:])


# In[ ]:





# ## Visualization with SpaCy

# In[3]:


import spacy
from spacy import displacy
from spacy.attrs import LOWER, POS, ENT_TYPE, IS_ALPHA
from spacy.tokens import Doc


# In[4]:


df_dev = pd.read_csv("transitional_data/tagged_dev_filled.csv", keep_default_na=False)


# In[15]:


print(df_dev.groupby("SentenceNR").get_group(0)["Label"].tolist())


# In[38]:


sents20 = []
labels20 = []
for i in range(20):
    
    sent = df_dev.groupby("SentenceNR").get_group(i)["Token"].tolist()
    sent.append("\n")
    sents20 += sent
    
    label = df_dev.groupby("SentenceNR").get_group(i)["Label"].tolist()
    label.append("o")
    labels20 += label


# In[39]:


len(sents) == len(labels)


# In[40]:


labels_new = []
for i in labels20:
    if i != "o":
        labels_new.append(i)
    else:
        labels_new.append("O")


# In[ ]:





# In[42]:


nlp = spacy.load("en_core_web_sm")
doc = Doc(nlp.vocab, words=sents20, ents = labels_new)


# In[43]:


colors = {'COURT': "#bbabf2", 'PETITIONER': "#f570ea", "RESPONDENT": "#cdee81", 'JUDGE': "#fdd8a5",
          "LAWYER": "#f9d380", 'WITNESS': "violet", "STATUTE": "#faea99", "PROVISION": "yellow",
          'CASE_NUMBER': "#fbb1cf", "PRECEDENT": "#fad6d6", 'DATE': "#b1ecf7", 'OTHER_PERSON': "#b0f6a2",
          'ORG':'#a57db5','GPE':'#7fdbd4'}


# In[44]:


options = {"colors": colors}
displacy.serve(doc, style='ent', options=options)


# In[ ]:




