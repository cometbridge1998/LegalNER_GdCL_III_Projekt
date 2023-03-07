#!/usr/bin/env python
# coding: utf-8

# # Feature Matrix
# Purpose of notebook is to enlarge the dataframe and provide the maschine learning model with more <b>context information</b>. <br>
# e. g. <i>Tokens on the left and right sides and its pos tags, lemmas.</i> <br>
# Because the tagger in last notebook already provides the pos tags for every token, the prefix, suffix and other features of the token are regarded as surplus and won't be included in the feature matrix.

# In[3]:


import pandas as pd
import numpy as np
import scipy as sp
import re


# In[1]:


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


# In[4]:


df_train = pd.read_csv("transitional_data/tagged_train_filled.csv", keep_default_na=False)
df_dev = pd.read_csv("transitional_data/tagged_dev_filled.csv", keep_default_na=False)


# <br> to check whether there are still empty cells in the dataframe

# In[5]:


df_train["Lemma"].isnull().tolist().count(True)


# In[6]:


df_train.dtypes


# In[7]:


df_train.Token = df_train.Token.astype("string")
df_train.Label = df_train.Label.astype("string")
df_train.standard_tagger = df_train.standard_tagger.astype("string")
df_train.TreeTagger = df_train.TreeTagger.astype("string")
df_train.Lemma = df_train.Lemma.astype("string")
df_train.dtypes


# In[8]:


df_dev.Token = df_dev.Token.astype("string")
df_dev.Label = df_dev.Label.astype("string")
df_dev.standard_tagger = df_dev.standard_tagger.astype("string")
df_dev.TreeTagger = df_dev.TreeTagger.astype("string")
df_dev.Lemma = df_dev.Lemma.astype("string")
df_dev.dtypes


# <br> number of sentences should start from 1

# In[9]:


df_train["SentenceNR"] = df_train["SentenceNR"].apply(lambda x: x+1)
df_dev["SentenceNR"] = df_dev["SentenceNR"].apply(lambda x: x+1)


# <br> rearrange the sequence of columns in the dataframe

# In[119]:


df_train["TokenNR"] = np.nan
df_train = df_train.rename(columns={"standard_tagger": "StandardTagger"})
df_train = df_train[['SentenceNR', 'TokenNR', 'Token', 'StandardTagger', 'TreeTagger', 'Lemma', 'Label']]
df_dev["TokenNR"] = np.nan
df_dev = df_dev.rename(columns={"standard_tagger": "StandardTagger"})
df_dev = df_dev[['SentenceNR', 'TokenNR', 'Token', 'StandardTagger', 'TreeTagger', 'Lemma', 'Label']]


# <br> with df.groupby(by = 'SentenceNR') the dataframe will be grouped according to the number of sentences.<br>
# And with the function enumerate_tokens the order of tokens in a sentence will also be supplemented to the dataframe.

# In[11]:


def enumerate_tokens(sentence):
    c = 1
    for index, row in sentence.iterrows():
        sentence.at[index, 'TokenNR'] = c
        c += 1
    return sentence


# In[12]:


get_ipython().run_line_magic('time', "df_dev = df_dev.groupby(by = 'SentenceNR', group_keys=True).apply(enumerate_tokens)")


# In[13]:


get_ipython().run_line_magic('time', "df_train = df_train.groupby(by = 'SentenceNR', group_keys=True).apply(enumerate_tokens)")


# In[14]:


df_train.TokenNR = df_train.TokenNR.astype("int64")
df_dev.TokenNR = df_dev.TokenNR.astype("int64")


# In[21]:


df_train = df_train.rename(columns={"standard_tagger": "StandardTagger"})
df_train = df_train[['SentenceNR', 'TokenNR', 'Token', 'StandardTagger', 'TreeTagger', 'Lemma', 'Label']]
df_dev = df_dev.rename(columns={"standard_tagger": "StandardTagger"})
df_dev = df_dev[['SentenceNR', 'TokenNR', 'Token', 'StandardTagger', 'TreeTagger', 'Lemma', 'Label']]


# <br>make copies of the train and dev dataframe so that the originals won't be changed in processing afterwards.

# In[23]:


train = df_train.copy()
dev = df_dev.copy()


# In[24]:


train = train.rename(columns={"SentenceNR": "Sent"})


# In[25]:


dev = dev.rename(columns={"SentenceNR": "Sent"})


# In[ ]:





# ## Initialize three CountVectorizers with  train.Token,  train.StandardTagger  and  train.Lemma
# The <b>wf-, tf-, lf_vectorizer</b> convert three columns "Token", "StandardTagger" and "Lemma" to sparse matrix as foundations of the bigger feature (sparse) matrixes in the following steps. <br>
# In the step of context information these three vectorizers will also be applied to the context tokens around the original token.

# In[57]:


# Token to spare Matrix
wf_vectorizer = CountVectorizer(tokenizer=lambda x: (x,), lowercase=False, min_df=3)
get_ipython().run_line_magic('time', 'train_X_wf = wf_vectorizer.fit_transform(train.Token)')
get_ipython().run_line_magic('time', 'dev_X_wf = wf_vectorizer.transform(dev.Token)')
print(train_X_wf.shape, dev_X_wf.shape)


# In[58]:


# Tag to spare Matrix
tf_vectorizer = CountVectorizer(tokenizer=lambda x: (x,), lowercase=False, min_df=3)
get_ipython().run_line_magic('time', 'train_X_tf = tf_vectorizer.fit_transform(train.StandardTagger)')
get_ipython().run_line_magic('time', 'dev_X_tf = tf_vectorizer.transform(dev.StandardTagger)')
print(train_X_tf.shape, dev_X_tf.shape)


# In[59]:


# Lemma to spare Matrix
lf_vectorizer = CountVectorizer(tokenizer=lambda x: (x,), lowercase=False, min_df=3)
get_ipython().run_line_magic('time', 'train_X_lf = lf_vectorizer.fit_transform(train.Lemma)')
get_ipython().run_line_magic('time', 'dev_X_lf = lf_vectorizer.transform(dev.Lemma)')
print(train_X_lf.shape, dev_X_lf.shape)


# In[ ]:





# ## Step 0
# ## Basis Matrix: only with Token, Tag and Lemma

# to compare with the classification results of "wider" matrixes (matrixes with more columns) in the following steps. <br>
# All steps will use the same model: the <b>default LinearSVC</b> by sklearn. <br><br>
# <i>Special Attention:</i> <br>
# The "outsider" ("o") tokens, with makes up around 84% percent of all tokens, will be <b>excluded</b> from the classfication report, so that the result can concentrate on the named entity labels in the dataset.

# ### Result: 
# ### weighted average for f1-score: 39% (dev), 44% (train)

# In[64]:


X_train = sp.sparse.hstack([train_X_wf, train_X_tf, train_X_lf])
X_dev = sp.sparse.hstack([dev_X_wf, dev_X_tf, dev_X_lf])
y_train = train["Label"]
y_dev = dev["Label"]


# In[65]:


classes = train["Label"].unique().tolist()
classes.remove("o")
print(classes)


# In[66]:


svc = LinearSVC()
get_ipython().run_line_magic('time', 'svc.fit(X_train, y_train)')


# In[67]:


get_ipython().run_cell_magic('time', '', 'y_dev_pred = svc.predict(X_dev)\nprint(classification_report(y_pred = y_dev_pred, y_true = y_dev, labels = classes))\n')


# In[70]:


get_ipython().run_cell_magic('time', '', 'y_train_pred = svc.predict(X_train)\nprint(classification_report(y_pred = y_train_pred, y_true = y_train, labels = classes))\n')


# In[ ]:





# ## Step 1
# ## Will the prefixes and suffixes contribute to the model? --A small increase

# ### Result: 
# ### weighted average for f1-score: 42% (dev), 50% (train), comparing to the basic model +3%, +6%

# ### get_prefix_suffix:
# return all prefixes and suffixes in a token from the length of 2 to 5.

# In[42]:


def get_prefix_suffix(word):
    l = len(word)
    res = []
    for k in range(2, 5):
        if l > k:
            res.append("-" + word[-k:])
    for k in range(2, 5):
        if l > k:
            res.append(word[:k] + "-")
    return(res)


# In[43]:


print(train.Token.tolist()[5], get_prefix_suffix(train.Token.tolist()[5]))
print(dev.Token.tolist()[6], get_prefix_suffix(dev.Token.tolist()[6]))


# <br> have a look at all affixes appearing more than 5000 times in the training dataset.

# In[45]:


affix_vectorizer = CountVectorizer(tokenizer=get_prefix_suffix, min_df=5000)
affix_vectorizer.fit(train.Token.tolist())
print(" ".join(affix_vectorizer.get_feature_names_out()))


# <br> For real usage we will set the min_df to a much lower number, to provide the model with more affix information. 

# In[71]:


affix_vectorizer = CountVectorizer(tokenizer=get_prefix_suffix, min_df=20)
train_X_affix = affix_vectorizer.fit_transform(train.Token.tolist())
dev_X_affix = affix_vectorizer.transform(dev.Token.tolist())


# In[72]:


X_train_affix.shape


# In[73]:


X_dev_affix.shape


# <br> with sp.sparse.hstack combine the X_train from the last step with the new affixes.

# In[75]:


X1_train = sp.sparse.hstack([X_train, train_X_affix])
X1_dev = sp.sparse.hstack([X_dev, dev_X_affix])


# In[77]:


clf = LinearSVC()
get_ipython().run_line_magic('time', 'clf.fit(X1_train, y_train)')


# In[78]:


get_ipython().run_cell_magic('time', '', 'y1_dev_pred = clf.predict(X1_dev)\nprint(classification_report(y_pred = y1_dev_pred, y_true = y_dev, labels = classes))\n')


# In[80]:


get_ipython().run_cell_magic('time', '', 'y1_train_pred = clf.predict(X1_train)\nprint(classification_report(y_pred = y1_train_pred, y_true = y_train, labels = classes))\n')


# In[ ]:





# ## Step 2
# ## Will other features of the tokens contribute to the model? --Nothing changes

# ### Result: 
# ### weighted average for f1-score: 42% (dev), 51% (train), comparing to the basic model +3%, +7%

# Supossedly because the matrix already have pos tags and lemmas, other features of the tokens cannot help the model to learn better. <br>
# Since they don't make a difference, the other features won't be used in future steps.

# In[84]:


def get_other_features(df, test=False):
    res = pd.DataFrame({
        'upper': df.Token.str.match(r'[A-Z]'),
        'allcaps': df.Token.str.fullmatch(r'[A-Z]+'),
        'digits': df.Token.str.match(r'[0-9]'),
        'alldigits': df.Token.str.fullmatch(r'-?[0-9][0-9.,]*'),
        'noalpha': ~df.Token.str.contains(r'[a-z]', flags=re.IGNORECASE),
        'noalnum': ~df.Token.str.contains(r'[0-9a-zäöü]', flags=re.IGNORECASE),
        'atstart': df.TokenNR == 1,
        'trunc': df.Token.str.endswith('-'),
        'long': df.Token.str.len() >= 15,
    })
    if test:
        return res
    else:
        return res.iloc[:, 1:].to_numpy(dtype=np.float64)


# In[85]:


get_ipython().run_line_magic('time', 'train_X_other = get_other_features(train)')
get_ipython().run_line_magic('time', 'dev_X_other = get_other_features(dev)')


# In[86]:


X2_train = sp.sparse.hstack([X_train, train_X_affix, train_X_other])
X2_dev = sp.sparse.hstack([X_dev, dev_X_affix, dev_X_other])


# In[88]:


svc = LinearSVC()
get_ipython().run_line_magic('time', 'svc.fit(X2_train, y_train)')


# In[89]:


get_ipython().run_cell_magic('time', '', 'y2_dev_pred = svc.predict(X2_dev)\nprint(classification_report(y_pred = y2_dev_pred, y_true = y_dev, labels = classes))\n')


# In[112]:


get_ipython().run_cell_magic('time', '', 'y2_train_pred = svc.predict(X2_train)\nprint(classification_report(y_pred = y2_train_pred, y_true = y_train, labels = classes))\n')


# In[ ]:





# ## Step 3
# ## Context left and right: A great difference!

# ### Result: 
# ### weighted average for f1-score: 77% (dev), 94% (train), comparing to the basic model +38%, +50%

# ## add_context
# At first we will add new columns to the both dataframes. <br>
# The new columns show the tokens, tags and lemmas in the rows before and after.<br>
# <i>(2 words on the left and 2 words on the right in the original text)</i><br> <br>
# This process will be executed at the level of <b>each sentence</b> because the sentences are disjunctive in the dataframe. <br>
# In other words, "neighbour" sentences don't belong to the same judgement. They are randomly mixed. <br>
# Beginnings and ends of all sentences will be padded. 

# In[91]:


def add_context(satz):
    
    satz["L1"] = satz.Token.shift(1, fill_value="")  
    satz["L2"] = satz.Token.shift(2, fill_value="")  
    satz["R1"] = satz.Token.shift(-1, fill_value="") 
    satz["R2"] = satz.Token.shift(-2, fill_value="") 
    
    satz["posL1"] = satz.StandardTagger.shift(1, fill_value="*")
    satz["posL2"] = satz.StandardTagger.shift(2, fill_value="*")
    satz["posR1"] = satz.StandardTagger.shift(-1, fill_value="*")
    satz["posR2"] = satz.StandardTagger.shift(-2, fill_value="*")
    
    satz["lemmaL1"] = satz.Lemma.shift(1, fill_value="*")
    satz["lemmaL2"] = satz.Lemma.shift(2, fill_value="*")
    satz["lemmaR1"] = satz.Lemma.shift(-1, fill_value="*")
    satz["lemmaR2"] = satz.Lemma.shift(-2, fill_value="*")
    
    # Labels of two tokens before are just preparation for the trigramme model 
    satz["labelL1"] = satz.Label.shift(1, fill_value="*")
    satz["labelL2"] = satz.Label.shift(2, fill_value="*")
    
    return satz


# In[92]:


get_ipython().run_line_magic('time', "train = train.groupby('Sent', group_keys=False).apply(add_context)")
get_ipython().run_line_magic('time', "dev = dev.groupby('Sent', group_keys=False).apply(add_context)")


# <br>Of course the last two columns ("LabelL1", "LabelL2")of dev are NOT allowed to be included in the feature matrix.<br>
# Otherwise they would lead to data leak.

# In[95]:


dev


# <br> Transform the tokens in the context with wf_vectorizer.

# In[98]:


get_ipython().run_cell_magic('time', '', 'train_token_context = sp.sparse.hstack([wf_vectorizer.transform(train.L1), \n                                        wf_vectorizer.transform(train.L2), \n                                        wf_vectorizer.transform(train.R1),\n                                        wf_vectorizer.transform(train.R2)])\n')


# In[104]:


train_token_context.shape


# <br> Transform the POS tags in the context with tf_vectorizer.

# In[97]:


get_ipython().run_cell_magic('time', '', 'train_tag_context = sp.sparse.hstack([tf_vectorizer.transform(train.posL1), \n                                        tf_vectorizer.transform(train.posL2), \n                                        tf_vectorizer.transform(train.posR1),\n                                        tf_vectorizer.transform(train.posR2)])\n')


# In[102]:


train_tag_context.shape


# <br> Transform the lemmas in the context with lf_vectorizer.

# In[103]:


get_ipython().run_cell_magic('time', '', 'train_lemma_context = sp.sparse.hstack([lf_vectorizer.transform(train.lemmaL1), \n                                        lf_vectorizer.transform(train.lemmaL2), \n                                        lf_vectorizer.transform(train.lemmaR1),\n                                        lf_vectorizer.transform(train.lemmaR2)])\n')


# In[105]:


train_lemma_context.shape


# <br> The same way for dev

# In[110]:


get_ipython().run_cell_magic('time', '', '\ndev_token_context = sp.sparse.hstack([wf_vectorizer.transform(dev.L1), \n                                        wf_vectorizer.transform(dev.L2), \n                                        wf_vectorizer.transform(dev.R1),\n                                        wf_vectorizer.transform(dev.R2)])\n\ndev_tag_context = sp.sparse.hstack([tf_vectorizer.transform(dev.posL1), \n                                        tf_vectorizer.transform(dev.posL2), \n                                        tf_vectorizer.transform(dev.posR1),\n                                        tf_vectorizer.transform(dev.posR2)])\n\ndev_lemma_context = sp.sparse.hstack([lf_vectorizer.transform(dev.lemmaL1), \n                                        lf_vectorizer.transform(dev.lemmaL2), \n                                        lf_vectorizer.transform(dev.lemmaR1),\n                                        lf_vectorizer.transform(dev.lemmaR2)])\n')


# In[109]:


X3_train = sp.sparse.hstack([X_train, train_token_context, train_tag_context, train_lemma_context])
X3_train.shape


# In[111]:


X3_dev = sp.sparse.hstack([X_dev, dev_token_context, dev_tag_context, dev_lemma_context])
X3_dev.shape


# <br> At first we will simply provide the model with the +-2 context and their tags and lemmas to see its effect alone.

# In[113]:


svc = LinearSVC()
get_ipython().run_line_magic('time', 'svc.fit(X3_train, y_train)')


# In[114]:


y3_dev_pred = svc.predict(X3_dev)
print(classification_report(y_pred = y3_dev_pred, y_true = y_dev, labels = classes))


# In[115]:


y3_train_pred = svc.predict(X3_train)
print(classification_report(y_pred = y3_train_pred, y_true = y_train, labels = classes))


# In[ ]:





# ## Step 4
# ## Context and Affix of context: Over training!

# ### Result: 
# ### weighted average for f1-score: 77% (dev), 97% (train), comparing to the basic model +38%, +53%

# Comparing to the feature matrix with context but without affix of the contextes (last step, X3), <br>
# providing the model also the affix of contextes just leads to <b>an increased over training</b>, <br>
# but it brings nothing in predicting the dev dataset. 

# In[120]:


get_ipython().run_cell_magic('time', '', 'train_affix_context = sp.sparse.hstack([affix_vectorizer.transform(train.L1.tolist()),\n                                       affix_vectorizer.transform(train.L2.tolist()),\n                                       affix_vectorizer.transform(train.R1.tolist()),\n                                       affix_vectorizer.transform(train.R2.tolist())])\n')


# In[121]:


get_ipython().run_cell_magic('time', '', 'dev_affix_context = sp.sparse.hstack([affix_vectorizer.transform(dev.L1.tolist()),\n                                       affix_vectorizer.transform(dev.L2.tolist()),\n                                       affix_vectorizer.transform(dev.R1.tolist()),\n                                       affix_vectorizer.transform(dev.R2.tolist())])\n')


# In[122]:


X4_train = sp.sparse.hstack([X_train, train_X_affix, train_token_context, train_tag_context, train_lemma_context, train_affix_context])
X4_dev = sp.sparse.hstack([X_dev, dev_X_affix, dev_token_context, dev_tag_context, dev_lemma_context, dev_affix_context])


# In[123]:


X4_train.shape


# In[124]:


X4_dev.shape


# In[125]:


svc = LinearSVC()
get_ipython().run_line_magic('time', 'svc.fit(X4_train, y_train)')


# In[126]:


get_ipython().run_cell_magic('time', '', 'y4_dev_pred = svc.predict(X4_dev)\nprint(classification_report(y_pred = y4_dev_pred, y_true = y_dev, labels = classes))\n')


# In[127]:


get_ipython().run_cell_magic('time', '', 'y4_train_pred = svc.predict(X4_train)\nprint(classification_report(y_pred = y4_train_pred, y_true = y_train, labels = classes))\n')


# In[ ]:





# ## Step 5
# ## Context and only Affix of token itself: No improve, but over training reduced
# ### Result: 
# ### weighted average for f1-score: 77% (dev), 95% (train), comparing to the basic model +38%, +51%

# This time we remove the affix of contextes from the feature matrix. <br>
# Although it doesn't bring a better score for the dev, <br>
# but it reduced a little bit the over training as last time. 

# In[128]:


X5_train = sp.sparse.hstack([X_train, train_X_affix, train_token_context, train_tag_context, train_lemma_context])
X5_dev = sp.sparse.hstack([X_dev, dev_X_affix, dev_token_context, dev_tag_context, dev_lemma_context])


# In[129]:


svc = LinearSVC()
get_ipython().run_line_magic('time', 'svc.fit(X5_train, y_train)')


# In[131]:


get_ipython().run_cell_magic('time', '', 'y5_dev_pred = svc.predict(X5_dev)\nprint(classification_report(y_pred = y5_dev_pred, y_true = y_dev, labels = classes))\n')


# In[132]:


get_ipython().run_cell_magic('time', '', 'y5_train_pred = svc.predict(X5_train)\nprint(classification_report(y_pred = y5_train_pred, y_true = y_train, labels = classes))\n')


# In[ ]:





# ## Trigrame Processing: Even much worse than the basis matrix!
# ### Result: 
# ### weighted average for f1-score: 25% (dev), comparing to the basic model -14%

# In[133]:


label_vectorizer = OneHotEncoder(handle_unknown = 'infrequent_if_exist', min_frequency=5)
tmp_train = np.vstack([train.labelL1, train.labelL2, train.labelL2 + " " + train.labelL1]).T
X_train_label = label_vectorizer.fit_transform(tmp_train)
X_train_label.shape


# In[134]:


X6_train = sp.sparse.hstack([X_train, X_train_label])


# In[135]:


get_ipython().run_cell_magic('time', '', 'clf = LinearSVC()\nclf.fit(X6_train, y_train)\n')


# In[136]:


X6_train.shape


# In[137]:


X_train_label.shape


# In[ ]:





# In[139]:


def get_features(satz):
    return sp.sparse.hstack([
        
        sp.sparse.hstack([wf_vectorizer.transform(satz.Token)]),
        tf_vectorizer.transform(satz.TreeTagger),
        lf_vectorizer.transform(satz.Lemma),
        
        #wf_vectorizer.transform(satz.L1),
        #wf_vectorizer.transform(satz.L2),
        #wf_vectorizer.transform(satz.R1),
        #wf_vectorizer.transform(satz.R2),
        
        #tf_vectorizer.transform(satz.posL1),
        #tf_vectorizer.transform(satz.posL2),
        #tf_vectorizer.transform(satz.posR1),
        #tf_vectorizer.transform(satz.posR2),
        
        #lf_vectorizer.transform(satz.lemmaL1),
        #lf_vectorizer.transform(satz.lemmaL2),
        #lf_vectorizer.transform(satz.lemmaR1),
        #lf_vectorizer.transform(satz.lemmaR2),
    ], format='csr')


# In[141]:


def tag_sentence(satz):
    n = satz.shape[0]
    X = get_features(satz) # Matrix der Oberflächenmerkmale
    tags = []
    p1 = p2 = "*"          # vorhergehende Labels
    for i in range(n):
        x1 = X[i, :]
        x2 = label_vectorizer.transform(np.array([[p1, p2, p2 + " " + p1]]))
        x = sp.sparse.hstack([x1, x2])
        tag = clf.predict(x)[0] # liefert NumPy-Array zurück
        tags.append(tag)
        p2, p1 = p1, tag
    return pd.Series(tags, index=satz.index, dtype='string')


# In[ ]:





# In[142]:


get_ipython().run_cell_magic('time', '', "predicted = dev.groupby('Sent').apply(tag_sentence)\n")


# In[143]:


print(classes)


# In[144]:


get_ipython().run_line_magic('time', 'print(classification_report(y_pred=predicted, y_true=y_dev, labels = classes))')


# In[ ]:





# ## with data leak

# If we provide the dev matrix with basis matrix and the two "gold" Labels before, it will bring a surprisingly good result (84%). <br>
# Of course this couln'd be allowed in the tast. 

# In[149]:


tmp_dev = np.vstack([dev.labelL1, dev.labelL2, dev.labelL2 + " " + dev.labelL1]).T
X_dev_label = label_vectorizer.transform(tmp_dev)
X_dev_label.shape


# In[153]:


X7_dev = sp.sparse.hstack([X_dev, X_dev_label])


# In[154]:


X7_dev


# In[155]:


get_ipython().run_line_magic('time', 'print(classification_report(y_pred=clf.predict(X7_dev), y_true=y_dev, labels = classes))')


# In[ ]:




