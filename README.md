# LegalNER_GdCL_III_Project
Coursework of the seminar Grundlagen der Computerlinguitik III at the University of Erlangen <br>
Wintersemester 2022-2023 <br>
Under the guidance of female professor <b>Stefanie Evert</b>, Lehrstuhl für Korpus- und Computerlinguistik<br>
The project is a shared task from the <i>SemEval-2023</i>, which dedicates to the recognition of entities in the legal documents. (Legal Named Entity Recognizer)<br>
[Link to the shared task](https://sites.google.com/view/legaleval/home?pli=1)<br><br>

## The Tokenizer.ipynb
(also see Tokenizer.py) <br> 
uses a TreebankWordTokenizer to convert the annotated judgement texts from the <b> javascript objects (json) </b> into <b> pandas dataframes </b> <br>
The dataframes are stored in <i> transitional_data/tokenized_train.csv </i> and <i> transitional_data/tokenized_dev.csv </i><br><br>

## The POS_Tagger.ipynb
(also see POS_Tagger.py) <br>
uses two POS taggers to provide dataframe with tags and lemmas of each token (a row in the dataframe). <br>
The first is the standard tokenizer (a <b>pretrained PerceptronTagger</b>) provided by <b>nltk</b>. <br>
The second is the <b>TreeTagger</b> configured with the <b>Penn treebank</b>. <br>
The TreeTagger provides also lemmas of each token to the dataframe. <br>
The extanded dataframes are stored in <i>"transitional_data/tagged_train_filled.csv"</i> and <i>"transitional_data/tagged_dev_filled.csv"</i>. The empty cells (np.NaN) in the dataframe are substituted with "0". <br><br>

## About the TreeTagger
[Link to the install instruction of TreeTagger](https://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/)<br>
[Tutorial to use the TreeTagger in python](https://treetaggerwrapper.readthedocs.io/en/latest/)<br>
I have installed the TreeTagger directly in the project folder under the name "TreeTagger". It includes not only a the TreeTaggerwrapper, but also is configured with the Penn treebank.<br><br>

## Feature_Matrix.ipynb
Purpose of notebook is to enlarge the dataframe and provide the maschine learning model with more <b>context information</b>. <br>
e. g. <i>Tokens on the left and right sides and its pos tags, lemmas.</i> <br>
Because the Treetagger in last notebook already provides the pos tags to every token, the prefix, suffix and other features of the token are generally proven to be surplus and won't be included in the final feature matrix.<br>
The "Trigramme" Processing (add the labels of last two tokens to the feature matrix, "gold" for the train and dynamically the predicted labels in the dev) <br>
only makes the model much worse. <br>
The latest result shows, feature matrix works best with following columns: <br>
<i>Token, POSTag and Lemma of the Token itself and also the three features of its L1, L2, R1, R2 neighbours.</i><br>
<i>WITHOUT any Affix, other features or labels before.</i><br>
### Best result: Weighed average f1 score of all labels 77% 
<i>(exclduing the "o", outsiders, which makes up 86% of all tokens)</i>
