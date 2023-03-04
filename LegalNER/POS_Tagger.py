#!/usr/bin/env python
# coding: utf-8

# # POS_Tagger

# In this Notebook two POS taggers will be adopted to add the Part of Speech types of each token to the dataframe. <br>
# The first is the standard tokenizer (a <i>pretrained PerceptronTagger</i>) provided by <i>nltk</i>. <br>
# The second is the <i>TreeTagger</i> configured with the <i>Penn treebank</i>. <br>
# The TreeTagger provides also lemmas of each token to the dataframe.

# Specific informations about installing and configuring the TreeTagger are included in the README.md of the Github Repository.

# In[ ]:


from nltk import pos_tag
from nltk.tag.perceptron import PerceptronTagger
import pandas as pd
import numpy as np


# In[2]:


from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report


# In[3]:


import treetaggerwrapper


# <br> load the tokenized train and dev dataframes 

# In[49]:


df_train = pd.read_csv("tokenized_train.csv")
df_dev = pd.read_csv("tokenized_dev.csv")


# In[50]:


df_train


# ## tag_by_SentenceNR
# tagging the tokens per sentence <br>
# The second parameter "tagger" of this function should be a tagger object. (here: standard tokenizer of nltk "pos_tag") <br>
# It return a list of tags with the same length as the dataframe, otherwise a ValueError would be raised.

# In[51]:


def tag_by_SentenceNR(df, tagger):
    all_tags = []
    sentence_numbers = df["SentenceNR"].unique()
    for nr in sentence_numbers:
        s = df[ df["SentenceNR"] == nr ]
        s["Token"] = s["Token"].astype("str")
        tags = [e[1] for e in tagger(s["Token"].tolist())]
        all_tags += tags
    if len(all_tags)== len(df):
        return all_tags
    else:
        raise ValueError


# In[52]:


get_ipython().run_cell_magic('time', '', 'df_train["standard_tagger"] = tag_by_SentenceNR(df_train, pos_tag)\ndf_dev["standard_tagger"]  = tag_by_SentenceNR(df_dev, pos_tag)\n')


# In[53]:


df_train


# ## tag_by_SentenceNR_with_PerceptronTagger
# Because the PerceptronTagger cannot be called through the parameter. <br>
# This time a new tagger object will be initialised in each loop.

# In[10]:


def tag_by_SentenceNR_with_PerceptronTagger(df): 
    all_tags = []
    sentence_numbers = df["SentenceNR"].unique()
    for nr in sentence_numbers:
        s = df[ df["SentenceNR"] == nr ]
        s["Token"] = s["Token"].astype("str")
        pretrain = PerceptronTagger()
        tags = [e[1] for e in pretrain.tag(s["Token"].tolist())]
        all_tags += tags
    if len(all_tags)== len(df):
        return all_tags
    else:
        raise ValueError


# In[11]:


get_ipython().run_cell_magic('time', '', 'df_train["PerceptronTagger"] = tag_by_SentenceNR_with_PerceptronTagger(df_train)\ndf_dev["PerceptronTagger"] = tag_by_SentenceNR_with_PerceptronTagger(df_dev)\n')


# In[12]:


df_train


# <br> compare the outcome of standard tagger and PerceptronTagger

# In[14]:


df_train[df_train['standard_tagger'] != df_train['PerceptronTagger']]
df_dev[df_dev['standard_tagger'] != df_dev['PerceptronTagger']]


# They are actually the same. <br>
# Delete the standard_tagger from the dataframe.

# In[15]:


df_train.drop(['standard_tagger'], axis=1)
df_dev.drop(['standard_tagger'], axis=1)


# In[ ]:





# ## tag_with_Treetagger
# The Treetagger always requires a continous str input, rather than a list of tokens. <br>
# <i>e. g.<br> 
#     tags = tagger.tag_text("This is a very short text to tag.")<br>
#     tags[0] = This  (tab)  DT  (tab)  this<br>
#     tags[1] = is  (tab)  VBZ  (tab)  be<br>
#     ...</i> 

# ## different tokenizing, how to cope with?
# Of course the TreeTagger will <b>NOT</b> produce the same the tokening outcome, since sentences rather than tokens are given. <br>
# Actually if tokens in list are given, the tagger will combine the tokens before performing the parsing and finally also return a different tokening output. <br>
# And since the TreebankWordTokenizer of nltk returns a reliable outcome according to the comparison with annotated labels,<br>
# we don't want to change the tokenizing.<br><br>
# Instead the tokens and tags returned from the TreeTagger will be compared to the tokens (rows) in the original dataframe. <br>
# For the "equivalent" tokens (who appear not only in the dataframe but also in the the TreeTagger outputs) their POS tags and lemmas will be added to the dataframe. <br>
# The different tokens will be simply dismissed. The "TreeTagger" and "lemma" elements in their rows will keep empty value "np.nan".

# In[45]:


def tag_with_Treetagger(df):
    
    # create a tagger object from treetaggerwrapper
    # TAGDIR="TreeTagger" shows where is the package installed.
    tagger = treetaggerwrapper.TreeTagger(TAGLANG='en', TAGDIR="TreeTagger")
    
    # create two empty columns in the dataframe
    df["TreeTagger"] = np.nan
    df["Lemma"] = np.nan
    
    # tagging by each sentence
    sentence_numbers = df["SentenceNR"].unique()
    
    for nr in sentence_numbers:
        s = df[ df["SentenceNR"] == nr ]
        tokens = [str(t) for t in s["Token"].tolist()]
        tags_and_lemmas = tagger.tag_text( " ".join( tokens ))
        penn = []
        for row in tags_and_lemmas:
            row = row.split("\t")
            if len(row) == 3:
                penn.append([ row[0], row[1], row[2] ])
            else:
                penn.append([np.nan, np.nan, np.nan])
        
        for unit in penn:
            for i in s.index.tolist():
                if str(unit[0]) == str(df.at[i, "Token"]) and pd.isnull(df.at[i, "TreeTagger"]) and pd.isnull(df.at[i, "Lemma"]):
                    df.at[i, "TreeTagger"] = unit[1]
                    df.at[i, "Lemma"] = unit[2]
                    break


# <br> Unluckily the comparing process is very inefficient and takes even much more time than tagging itself.

# In[54]:


get_ipython().run_cell_magic('time', '', 'tag_with_Treetagger(df_train)\n')


# In[55]:


get_ipython().run_cell_magic('time', '', 'tag_with_Treetagger(df_dev)\n')


# In[56]:


df_train


# In[57]:


df_dev


# <br> How many cells in the dataframe are still empty?

# In[64]:


for column in df_train.columns:
    print(df_train[column].isnull().value_counts())


# <br> 2.28 % of the tokens didn't become the TreeTagger Tag and Lemma.

# In[65]:


for column in df_dev.columns:
    print(df_dev[column].isnull().value_counts())


# ## Fill the NaNs
# With .fillna() the empty cells in the dataframe will be substituted.<br>
# This is necessary for the preparing of the maschine learning.

# In[66]:


df_train_filled = df_train.copy()
df_train_filled = df_train_filled.fillna("0")


# In[67]:


df_dev_filled = df_train.copy()
df_dev_filled = df_train_filled.fillna("0")


# In[58]:


df_train.to_csv("tagged_train.csv", index=False)


# In[59]:


df_dev.to_csv("tagged_dev.csv", index=False)


# In[82]:


df_train_filled.to_csv("tagged_train_filled.csv", index=False)


# In[83]:


df_dev_filled.to_csv("tagged_dev_filled.csv", index=False)


# <br> Check one more time<br>There is no more NaNs in the dataframe

# In[73]:


for column in df_train_filled.columns:
    print(df_train_filled[column].isnull().value_counts())


# In[ ]:





# In[ ]:





# In[ ]:





# ## Do the tags and lemmas make a difference? YES!
# This time with the same primitive model and parameter the weighted avg reached  <b> 37% </b>. <br>
# It has improved <b>9%</b> comparing to the "thin" matrix with only the tokens itself last time.

# In[76]:


X_train = df_train_filled.drop(["Label", "SentenceNR"], axis = 1)
v = DictVectorizer(sparse=True)
X_train = v.fit_transform(X_train.to_dict('records'))
y_train = df_train_filled["Label"]

X_dev = df_dev_filled.drop(["Label", "SentenceNR"], axis=1)
X_dev = v.transform(X_dev.to_dict('records'))
y_dev = df_dev_filled["Label"]

print(X_train.shape, y_train.shape)
print(X_dev.shape, y_dev.shape)


# In[77]:


classes = df_train_filled["Label"].unique().tolist()
print(classes)


# In[78]:


per = Perceptron(verbose=10, n_jobs=-1, max_iter=5)
per.partial_fit(X_train, y_train, classes)


# In[79]:


classes.remove("o")
print(classification_report(y_pred=per.predict(X_dev), y_true=y_dev, labels=classes))


# In[ ]:





# In[ ]:





# In[ ]:




