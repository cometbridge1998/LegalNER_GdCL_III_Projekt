# LegalNER_GdCL_III_Project
Coursework of the seminar Grundlagen der Computerlinguitik III at the University of Erlangen <br>
Wintersemester 2022-2023 <br>
Under the guidance of female professor <b>Stefanie Evert</b>, Lehrstuhl für Korpus- und Computerlinguistik<br>
The project is a shared task from the <i>SemEval-2023</i>, which dedicates to the recognition of entities in the legal documents. (Legal Named Entity Recognizer)<br>
[Link to the shared task](https://sites.google.com/view/legaleval/home?pli=1)<br><br>

## The Tokenizer.ipynb
(also see Tokenizer.py) <br> 
uses a TreebankWordTokenizer to convert the annotated judgement texts from the <b> javascript objects (json) </b> into <b> pandas dataframes </b> <br>
The dataframes are stored in <i> transitional_data/tokenized_train.csv </i> and <i> transitional_data/tokenized_dev.csv </i>

## The POS_Tagger.ipynb
(also see POS_Tagger.py) <br>
uses two POS taggers to provide dataframe with tags and lemmas of each token (a row in the dataframe). <br>
The first is the standard tokenizer (a <b>pretrained PerceptronTagger</b>) provided by <b>nltk</b>. <br>
The second is the <b>TreeTagger</b> configured with the <b>Penn treebank</b>. <br>
The TreeTagger provides also lemmas of each token to the dataframe. <br>
The extanded dataframes are stored in <i>"transitional_data/tagged_train_filled.csv"</i> and <i>"transitional_data/tagged_dev_filled.csv"</i>. The empty cells (np.NaN) in the dataframe are substituted with "0". <br>

## About the TreeTagger
[Link to the install instruction of TreeTagger](https://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/)<br>
[Tutorial to use the TreeTagger in python](https://treetaggerwrapper.readthedocs.io/en/latest/)<br>
I have installed the TreeTagger directly in the project folder under the name "TreeTagger". It includes not only a the TreeTaggerwrapper, but also is configured with the Penn treebank.
