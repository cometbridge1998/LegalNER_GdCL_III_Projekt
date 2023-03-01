#!/usr/bin/env python
# coding: utf-8

# # Tokenizing
# <br>
# Purpose of this notebook is to convert the annotated judgement texts from the <b> javascript objects (json) </b> into <b> pandas dataframes </b>, which can be used as the matrix for mashine learning.<br> <br>
# Labels in the annotated texts are stored in the json trees according to the sequence number of characters. <br>
# <i> e.g. " ... Hongkong Bank ... " - { 'value': {'start': 90, 'end': 103, 'text': 'Hongkong Bank','labels': ["ORG"] } </i> <br> <br> 
# With the <b> span_tokenize </b> the labels will be adapted to the sequence of tokens. <br>
# <i> e.g. [ ... 'B-ORG', 'I-ORG', ... ] </i> <br> <br> 
# Each token and its label makes up a single row in the dataframe.

# In[ ]:





# In[1]:


import pandas as pd
import numpy as np
import json


# In[2]:


from nltk.tokenize import word_tokenize, TreebankWordTokenizer, TreebankWordDetokenizer


# In[3]:


from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report


# In[4]:


# read the training dataset
with open("NER_TRAIN/NER_TRAIN_JUDGEMENT.json") as json_file_train:
    json_object_train = json.load(json_file_train)


# In[5]:


# read the developing dataset
with open("NER_DEV/NER_DEV/NER_DEV_JUDGEMENT.json") as json_file_dev:
    json_object_dev = json.load(json_file_dev)


# In[ ]:





# <br>
# Have a look at th first sentence of the judgment. (the first json tree) <br>
# Each tree includes only one sentence.

# In[6]:


json_object_train[0]


# In[ ]:





# ### get_start_and_end_and_labels
# This function is designed to extract all labels with their spans from a json tree. <br>
# It returns a list of labels. (length of the list corresponds to the number of total labels in the sentence. A label that spans more than one tokens count also as only one label) <br>
# Each label is list of three elements.<br>
# The first element is the name of the label, e. g. "ORG". <br>
# The second element is the start of label, the third the end of the label.

# In[7]:


def get_start_and_end_and_labels(tree):
    start_and_end_and_labels = []
    for label in tree["annotations"][0]["result"]:
        labels = label["value"]["labels"][0]
        start = label["value"]["start"]
        end = label["value"]["end"]
        start_and_end_and_labels.append([labels, start, end])
    return start_and_end_and_labels


# <br>
# The labels in the first sentence.

# In[8]:


get_start_and_end_and_labels(json_object_train[0])


# In[ ]:





# ### return_text_and_label
# returns a tuple in length of two. <br>
# The first element is the text of the tree. (str) <br>
# The second a list of all labels. (list)

# In[9]:


def return_text_and_label(tree):
    text = tree["data"]["text"]
    labels = get_start_and_end_and_labels(tree)
    return text, labels


# In[10]:


print(return_text_and_label(json_object_train[0]))


# In[ ]:





# ### Have a try of the TreebankWordTokenizer of nltk
# With TreebankWordDetokenizer the tokens stored in a list after tokenizing will be combined again to a str. 

# In[11]:


twt = TreebankWordTokenizer()
twd = TreebankWordDetokenizer()
try_text, try_label = return_text_and_label(json_object_train[0])
tokens = twt.tokenize(try_text)
print(f"tokenized text: \n{tokens}\n")
print(f"recombined text with detokenizer: \n{twd.detokenize(tokens)}")


# ### span_tokenize
# span_tokenize is a special function provided by nltk. <br>
# It return the start and end of each token (according to the number sequence of characters) in a list. <br>
# It is especially useful to 

# In[12]:


try_tokens = list(TreebankWordTokenizer().span_tokenize(try_text))
print(try_tokens)


# In[ ]:





# ### add_label_to_tokens
# This function is crucial in the processing of the raw data. <br>
# In the tradition of a NER (Named Entity Recognition) task, tokens should be tabbed not only by their labels, but also by their positions in the label. <br>
# When a token does not lay in any label, it should be tabbed als <b> "o" </b>. ("outsider") <br>
# When it lays at the beginning of a label, then <b> "B-" </b> ("beginning") plus the name of the label. <br>
# All tokens after the first token in a label should be tabbed as "insider". (<b> "I-" </b>) <br> 
# 
# <i> e. g. <br>
# " ... of Hongkong Bank ... " <br>
# ["o", "B-ORG", "I-ORG"]</i> <br> <br>
# 
# <i> special attention:</i> <br>
# The first parameter of this function "<b>tokens</b>" requires a list of <b>token spans</b>, which are the products of a span_tokenizer. <br> <br>
# 
# <i> maximal span strategy </i> <br>
# Because there is no guarantee that the tokenizer could always produce the same tokenizing as the one used by annotation. <br> 
# And it is also possible, that some labels of the annotation does not correspond to the boundaries of (natural) tokens because of carelessness or different understanding of the boundaries. <br>
# The maximal span strategy maximizes the included tokens in a label. So lang as a single character in the token is included in the label span, it will be labelled. <br> <br>
#     
# Later the quality of tokenizing and maximal span strategy will be checked with the <b>compare_label_with_labelled_tokens</b> function. <br>
# It will prove that the maximal span strategy would not almost change a single label.

# In[13]:


def add_label_to_tokens(tokens, labels):
    # at first, create a list of "o"s with the same length as the number of tokens in the sentence.
    # Hier the parameter tokens requires a list of token spans, which are the products of a span_tokenizer.
    token_labels = ["o" for token in tokens]
    
    # afterwards, search the tokens inside of the labels.
    # This process is not very efficient.
    # For each label in all labels it will interate all token spans in the sentence to find out which tokens belongs to this label.
    for label in labels:
        # label_start, label_end with character numbers
        label_start = label[1]
        label_end = label[2]
        if label_start <= label_end:
            for i in range(0, len(tokens)):
                # token_start, token_end also with character numbers
                token_start, token_end = tokens[i]
                
                # the first token in the label ("Beginning")
                if token_start <= label_start < token_end:
                    token_labels[i] = "B-" + label[0]
                
                # the last token in a label, if the label span does not correspond to the end of the token
                elif token_start < label_end <= token_end:
                    token_labels[i] = "I-" + label[0]
                
                # the following tokens after the first label ("Insider")
                if label_start < token_start <=  token_end <= label_end:
                    token_labels[i] = "I-" + label[0]
    return token_labels


# In[ ]:





# ### get_tokens_with_label
# This function is an expansion of <i>add_label_to_tokens</i>. <br>
# It wraps the <i>add_label_to_tokens</i> and provides it with the required parameters. <br>
# Besides it returns also a list of tokens from the tokenizer. (with characters, rather in spans)

# In[14]:


def get_tokens_with_label(tree):
    text, labels = return_text_and_label(tree)
    twt = TreebankWordTokenizer()
    tokens_span = list(TreebankWordTokenizer().span_tokenize(text))
    list_of_tokenized_text = twt.tokenize(text)
    list_of_label_of_each_token = add_label_to_tokens(tokens_span, labels)
    
    """
    d = {}
    for i in range(len(tokens_span)):
        d[ tokens_span[i] ] = tokenized_text[i]
    for key, value in d.items():
        print(f"{key}: {value}")
    """
    
    return  list_of_label_of_each_token, list_of_tokenized_text


# <br> print out the labels and tokens in the first tree with <i> get_tokens_with_label </i>

# In[15]:


list_of_label_of_each_token, list_of_tokens = get_tokens_with_label(json_object_train[0])
print(f"list_of_label_of_each_token: \n{list_of_label_of_each_token}\n")
print(f"list_of_tokenized_text: \n{list_of_tokens}")


# In[ ]:





# ### compare_label_with_labelled_tokens
# The functions compares the labelles tokens with the labelled text from the annotation to check the quality of Tokenizing. <br>

# In[16]:


def compare_label_with_labelled_tokens(tree, object_number, print_the_differences = False):
    number_of_errors = 0
    error_report = ""
    
    # labels_with_text : a list of named entities directly extracted from the text according to the annotation.
    text, labels = return_text_and_label(tree)
    labels_with_text = []
    for label in labels:
        labels_with_text.append(text[label[1]:label[2]])
    
    # all_labelled_tokens: a list of labelled texts after tokenizing with the maximize span strategy.
    labelled_tokens, tokenized_text = get_tokens_with_label(tree)
    all_labelled_tokens = []
    l = len(labelled_tokens)
    for i in range(l):
        single_label = []
        if labelled_tokens[i].startswith("B"):
            single_label.append(tokenized_text[i])
            while i+ 1 < l and labelled_tokens[i+1].startswith("I"):
                single_label.append(tokenized_text[i+1])
                i += 1
        if len(single_label) > 0:
            all_labelled_tokens.append(" ".join(single_label))
    
    # compare whether the length of two lists (number of labels in a sentence) are the same.
    if len(labels_with_text) != len(all_labelled_tokens):
        number_of_errors += abs( len(labels_with_text) - len(all_labelled_tokens) )
        error_report += "--------------\n"
        error_report += f"different number of labels at the {object_number}th object!\n"
        error_report += f"labels: {labels}\n"
        error_report += f"labels_with_text: {labels_with_text}\n"
        error_report += f"all_labelled_tokens: {all_labelled_tokens} \n"
        error_report += "--------------\n"
    
    # compare each element in both lists.
    else:
        for i in range(len(labels_with_text)):
            gold = labels_with_text[i].replace(" ", "")
            tokenized = labels_with_text[i].replace(" ", "")
            if gold != tokenized:
                number_of_errors += 1
                error_report += "--------------\n"
                error_report += f"different labels at the {object_number}th object!\n"
                error_report += "potential tokenizing problem: "
                error_report += f"gold: {gold} -- tokenized: {tokenized}"
                error_report += "--------------"
    
    if print_the_differences:
        print(error_report)
    
    return number_of_errors, error_report


# <br>
# compare the annotation and labelled texts after tokenizing in a single object (1328)

# In[17]:


number_of_errors_1328, error_report_1328 = compare_label_with_labelled_tokens(json_object_train[1328], object_number=1328, print_the_differences = True)


# In[18]:


number_of_errors_1328


# Analysis:<br>
# a tokenizing error at the end of the sentence: <br>
# annoatation: 'Koramangala P.S.Cr', '.', 'No.430/05'
# tokenized text : 'Koramangala', 'P.S.Cr.No.430/05'

# In[ ]:





# ### all wrong labelled entities after tokenizing in the training and development dataset

# In[19]:


with open("tokenizing_report_train.txt", "w", encoding="utf-8") as f:
    number_of_errors = 0
    error_report = ""
    for i in range(len(json_object_train)):
        new_errors, new_error_report = compare_label_with_labelled_tokens(json_object_train[i], object_number = i)
        number_of_errors += new_errors
        error_report += new_error_report
    f.write(error_report)
    print(f"total number of wrong tokenized labels in the training dataset: {number_of_errors}")
f.close()


# In[20]:


with open("tokenizing_report_dev.txt", "w", encoding="utf-8") as d:
    number_of_errors = 0
    error_report = ""
    for i in range(len(json_object_dev)):
        new_errors, new_error_report = compare_label_with_labelled_tokens(json_object_dev[i], object_number = i)
        number_of_errors += new_errors
        error_report += new_error_report
    d.write(error_report)
    print(f"total number of wrong tokenized labels in the developing dataset: {number_of_errors}")
d.close()


# ## Quality Analysis of the tokenizing
# The total number of wrong tokenized labels in both training and developing dataset are very low. <br>
# This proves the gut quality of tokenizing. <br>
# Among the 57966 labels in the training dataset are only 5 of them false tokenized. <br>
# (Tokenizing accuracy = 99.9914 %) 
# 
# 4 of the total 6 errors are caused by the dashes in the names. <br>
# <i>labels_with_text: [ ... 'Bangalore', 'Madras'] <br>
# all_labelled_tokens: [ .. 'Bangalore-Madras'] </i>

# In[ ]:





# ## Convert the json to dataframe
# As preparation of the POS-tagging in next step, the sentence number of each token will also be stored in the dataframe

# In[21]:


def json_to_df(trees):
    token_and_labels = []
    for n in range(len(trees)):
        labels, tokens = get_tokens_with_label(trees[n])
        if len(labels) != len(tokens):
            raise ValueError
        else:
            for i in range(len(labels)):
                token_and_labels.append([ n, tokens[i], labels[i]])
    df = pd.DataFrame(token_and_labels)
    df.columns = ['SentenceNR', 'Token', 'Label']
    return df


# In[22]:


df_train = json_to_df(json_object_train)
df_train


# In[23]:


df_dev = json_to_df(json_object_dev)
df_dev


# In[ ]:





# ## How good will it work right now?
# With a simple <i>Perceptron</i> modell from <i>Sklearn</i>

# In[45]:


X_train = df_train.drop(["Label", "SentenceNR"], axis = 1)
v = DictVectorizer(sparse=True)
X_train = v.fit_transform(X_train.to_dict('records'))
y_train = df_train["Label"]

X_dev = df_dev.drop(["Label", "SentenceNR"], axis=1)
X_dev = v.transform(X_dev.to_dict('records'))
y_dev = df_dev["Label"]

print(X_train.shape, y_train.shape)
print(X_dev.shape, y_dev.shape)


# In[46]:


classes = df_train["Label"].unique().tolist()
print(classes)


# In[47]:


per = Perceptron(verbose=10, n_jobs=-1, max_iter=5)
per.partial_fit(X_train, y_train, classes)


# In[48]:


classes.remove("o")
print(classification_report(y_pred=per.predict(X_dev), y_true=y_dev, labels=classes))


# In[ ]:





# In[ ]:




