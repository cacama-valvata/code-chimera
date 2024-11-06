# -*- coding: utf-8 -*-
"""TrueFoundary.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1-mu-R1CWen6FBdKrwbIitm1-j1vchBHf

# Loading the Data
"""

import warnings
warnings.simplefilter('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#nltk
import string # used for preprocessing
import re # used for preprocessing
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords # used for preprocessing
from nltk.stem import WordNetLemmatizer # used for preprocessing
from nltk.probability import FreqDist #fr frquency count
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import pickle
from tqdm import tqdm
import os

#preprocessing
from sklearn.feature_extraction.text import TfidfTransformer,TfidfVectorizer,CountVectorizer
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,OrdinalEncoder,PowerTransformer,StandardScaler,normalize
from sklearn.model_selection import train_test_split, StratifiedKFold,KFold,TimeSeriesSplit,cross_val_score,GridSearchCV
from sklearn.feature_selection import RFE
from sklearn.neighbors import LocalOutlierFactor
from sklearn.impute import SimpleImputer

# Classifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import LinearSVC,SVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, VotingClassifier
import xgboost as xgb
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#pip install lgm
from lightgbm import LGBMClassifier

# Metrics
from sklearn.metrics import roc_auc_score,make_scorer,classification_report,confusion_matrix,accuracy_score
from sklearn.multiclass import OneVsRestClassifier
import random
import torch

from google.colab import drive
drive.mount('/content/drive')

data = pd.read_csv('/content/drive/MyDrive/TrueFoundry/airline_sentiment_analysis.csv')
data

data.columns

# Rename the "old_name" column to "new_name"
data = data.rename(columns={'Unnamed: 0': 'id'})
data.head()

data['airline_sentiment'] = data['airline_sentiment'].replace({'positive': 1, 'negative': 0})
data.head()

import torch

if torch.cuda.is_available():       
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

"""# Preprocessing"""

import nltk
# Uncomment to download "stopwords"
nltk.download("stopwords")
from nltk.corpus import stopwords

def text_preprocessing(s):
    """
    - Lowercase the sentence
    - Change "'t" to "not"
    - Remove "@name"
    - Isolate and remove punctuations except "?"
    - Remove other special characters
    - Remove stop words except "not" and "can"
    - Remove trailing whitespace
    """
    s = s.lower()
    # Change 't to 'not'
    s = re.sub(r"\'t", " not", s)
    # Remove @name
    s = re.sub(r'(@.*?)[\s]', ' ', s)
    # Isolate and remove punctuations except '?'
    s = re.sub(r'([\'\"\.\(\)\!\?\\\/\,])', r' \1 ', s)
    s = re.sub(r'[^\w\s\?]', ' ', s)
    # Remove HTML encoding for < and >
    s = re.sub(r"&lt;", "<", s)
    s = re.sub(r"&gt;", ">", s)
    # Correct errors (eg. '&amp;' to '&')
    # Replace '&amp;' with '&'
    s = re.sub(r'&amp;', '&', s)
    # Remove some special characters
    s = re.sub(r'([\;\:\|•«\n])', ' ', s)
    # Remove stopwords except 'not' and 'can'
    '''
    s = " ".join([word for word in s.split()
                  if word not in stopwords.words('english')
                  or word in ['not', 'can']])
    '''
    # Remove special characters
    s = re.sub(r"[^a-zA-Z0-9]", " ", s)
    # Remove trailing whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    
    return s

t = data.text[1342]
t

# Print sentence 0
print('Original: ', t)
print('Processed: ', text_preprocessing(t))

"""----------------------------

just doing some dry run to verify some queries....
"""

# preprocessing "@AmericanAir right on cue with the delays👌".
'''
import nltk
import re

def preprocess_text(text):
    # Remove Twitter handle
    text = re.sub(r"@\w+", "", text)

    # Remove special characters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)

    # Convert to lowercase
    text = text.lower()

    # Remove excess white space
    text = re.sub(r"\s+", " ", text)

    return text.strip()

text = "@AmericanAir right on cue with the delays👌"
cleaned_text = preprocess_text(text)
print(cleaned_text)  # Output: "right on cue with the delays"

'''

"""continued.....

---------------------------
"""

t = data.airline_sentiment.values
t

"""# EDA

[link text](https://www.kaggle.com/code/ananysharma/sentiment-analysis-using-bert)
"""

data.airline_sentiment.value_counts()

plt.rcParams['figure.figsize'] = (5,5)
sns.countplot(data["airline_sentiment"],hue = data["airline_sentiment"],palette = 'dark')
plt.legend(loc = 'upper right')
plt.show()

train_temp = data.copy(deep=True)

unprocessed_X = list(train_temp["text"])
Y = list(train_temp["airline_sentiment"])

preprocessed_X = np.array([text_preprocessing(text) for text in unprocessed_X])

train = pd.DataFrame(list(zip(preprocessed_X, Y)), columns = ['text_a', 'labels'])
train.head()

"""**Analysing the words in sentences**"""

text_len = []
for text in train.text_a:
    tweet_len = len(text.split())
    text_len.append(tweet_len)

#train["text_len"] = train["text_a"].apply(lambda x: len(x)) || Do No Us
# _ t e

train['text_len'] = text_len

train.head()

lengths = pd.Series(sorted(train["text_len"]))
lengthsCount = pd.Series(sorted(train["text_len"])).value_counts()
lengths.plot()

plt.plot(lengthsCount,lengthsCount.index)

lengths.median()# Max number of sentences are of ~20 words.

lengths.describe()

train["text_a"].str.split().\
    map(lambda x: len(x)).\
    hist()

"""**Analysing the characters in sentences**"""

train["text_character_len"] = train["text_a"].apply(lambda x: len(x))

train.shape

train.head()

lengths = pd.Series(sorted(train["text_character_len"]))
lengthsCount = pd.Series(sorted(train["text_character_len"])).value_counts()
lengths.plot()

plt.plot(lengthsCount,lengthsCount.index)

lengths.median()# Max number of sentences are of ~104 words.

lengths.describe()

"""------------------------------

* Here also most of the examples are less than 150 charector.
* That means a 250 word truncation will not be a problem for this data currently. Since we have a max length of 34 words in a sentence.

-----------------------------
"""

eval = pd.read_csv("/content/drive/MyDrive/TrueFoundry/eval_dataframe.csv")

mydf = pd.read_csv("/content/drive/MyDrive/TrueFoundry/mydf_dataframe.csv")

!pip install transformers

"""# BERT"""

train_bert = data.copy(deep=True)

train_bert = train_bert.rename(columns={'airline_sentiment': 'labels'})

train_bert['text'] = train_bert['text'].apply(text_preprocessing)

text_len = []
for text in train_bert.text:
    tweet_len = len(text.split())
    text_len.append(tweet_len)

"""adding column for every text length"""

train_bert['text_len'] = text_len

# Importing stock ml libraries
import numpy as np
import pandas as pd
from sklearn import metrics
import transformers
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel, BertConfig


import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

#data processing
import re, string
import nltk

from sklearn import preprocessing
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split


#Naive Bayes
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

#transformers
from transformers import BertTokenizerFast
from transformers import TFBertModel
from transformers import RobertaTokenizerFast
from transformers import TFRobertaModel

#keras
import tensorflow as tf
from tensorflow import keras


#metrics
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix

#set seed for reproducibility
seed=42

#set style for plots
sns.set_style("whitegrid")
sns.despine()
plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True)
plt.rc("axes", labelweight="bold", labelsize="large", titleweight="bold", titlepad=10)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

token_lens = []

for txt in train_bert['text'].values:
    tokens = tokenizer.encode(txt, max_length=512, truncation=True)
    token_lens.append(len(tokens))
    
max_len=np.max(token_lens)

print(f"MAX TOKENIZED SENTENCE LENGTH: {max_len}")

"""adding column for every token length"""

train_bert['token_lens'] = token_lens

train_bert.head()

!pip install -U sentence-transformers

!pip install simpletransformers

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from simpletransformers.classification import (ClassificationModel, ClassificationArgs)
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

import torch

if torch.cuda.is_available():       
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

train_new = data.copy(deep=True)

train_new = train_new.iloc[np.random.permutation(len(train_new))] #shuffle
train_new = train_new.reset_index(drop=True)

train_new['airline_sentiment'] = train_new['airline_sentiment'].astype(float)

train_new.rename(columns={'text': 'text_a',
                          'airline_sentiment': 'labels'},
                           inplace=True, errors='raise')
train_new.head()

#applying the preprocessing function
train_new['text_a'] = train_new['text_a'].apply(text_preprocessing)

X = list(train_new["text_a"])
y = list(train_new["labels"])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, stratify = y)

mydf = pd.DataFrame(list(zip(X_train, y_train)), columns = ['text_a', 'labels'])
eval = pd.DataFrame(list(zip(X_val, y_val)), columns = ['text_a', 'labels'])

mydf.to_csv (r'/content/drive/MyDrive/TrueFoundry/mydf_dataframe.csv', index = None, header=True) 
eval.to_csv (r'/content/drive/MyDrive/TrueFoundry/eval_dataframe.csv', index = None, header=True)

mydf.labels.value_counts()

eval.labels.value_counts()

# Optional model configuration
model_args = ClassificationArgs(num_train_epochs=3)

# Create a ClassificationModel
model = ClassificationModel("bert","bert-base-cased")

from simpletransformers.classification import ClassificationModel, ClassificationArgs
model_args = ClassificationArgs(num_train_epochs=5,overwrite_output_dir=True,train_batch_size=16,max_seq_length = 128,output_dir ="/content/drive/MyDrive/TrueFoundry/BERT_model")
# Create a ClassificationModel
model_1 = ClassificationModel(
     "bert","bert-base-cased",
    # "distilbert","distilbert-base-cased",
    #"roberta", "roberta-base",
    num_labels=2,
    args=model_args)
#df_train = df_train[["text_filt","aspect","label"]]
#df_train.columns = ["text_a", "text_b", "labels"]
model_1.train_model(mydf)

pred,pred_prob = model_1.predict(eval[["text_a"]].values.tolist())

pd.Series(pred).value_counts()

eval["pred_bert"] = pred

print(classification_report(eval["labels"],eval["pred_bert"]))

print("final-f1 = ",f1_score(eval["labels"],eval["pred_bert"]))

from sklearn.metrics import matthews_corrcoef
print("the  Matthews correlation coefficient (MCC) score is =", matthews_corrcoef(eval["labels"],eval["pred_bert"]))

"""--------------
--------------

* **the final f1 score comes out to be 0.869**

* **the  Matthews correlation coefficient (MCC) score comes out to be 0.838**

--------------
--------------

# RoBERTa
"""

!pip install -U sentence-transformers

!pip install simpletransformers

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from simpletransformers.classification import (ClassificationModel, ClassificationArgs)
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

import torch

if torch.cuda.is_available():       
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

train_new = data.copy(deep=True)

train_new.head()

train_new = train_new.iloc[np.random.permutation(len(train_new))] #shuffle
train_new = train_new.reset_index(drop=True)

train_new.head()

train_new['airline_sentiment'] = train_new['airline_sentiment'].astype(float)

train_new.rename(columns={'text': 'text_a',
                          'airline_sentiment': 'labels'},
                           inplace=True, errors='raise')
train_new.head()

X = list(train_new["text_a"])
y = list(train_new["labels"])

X_preprocess = np.array([text_preprocessing(text) for text in X])

X_train, X_val, y_train, y_val = train_test_split(X_preprocess, y, test_size = 0.2, stratify = y)

#https://datascience.stackexchange.com/questions/40584/meaning-of-stratify-parameter

mydf = pd.DataFrame(list(zip(X_train, y_train)), columns = ['text_a', 'labels'])
mydf.head()

mydf.labels.value_counts()

eval = pd.DataFrame(list(zip(X_val, y_val)), columns = ['text_a', 'labels'])
eval.head()

eval.labels.value_counts()

# Optional model configuration
model_args = ClassificationArgs(num_train_epochs=3)

# Create a ClassificationModel
model = ClassificationModel("roberta", "roberta-base")

from simpletransformers.classification import ClassificationModel, ClassificationArgs
model_args = ClassificationArgs(num_train_epochs=5,overwrite_output_dir=True,train_batch_size=16,max_seq_length = 128,output_dir ="/content/drive/MyDrive/TrueFoundry/RoBERTa_model")
# Create a ClassificationModel
model_2 = ClassificationModel(
    # "bert","bert-base-cased",
    # "distilbert","distilbert-base-cased",
    "roberta", "roberta-base",
    num_labels=2,
    args=model_args)
#df_train = df_train[["text_filt","aspect","label"]]
#df_train.columns = ["text_a", "text_b", "labels"]
model_2.train_model(mydf)

#https://stackoverflow.com/questions/66600362/runtimeerror-cuda-error-cublas-status-execution-failed-when-calling-cublassge

pred,pred_prob = model_2.predict(eval[["text_a"]].values.tolist())

pd.Series(pred).value_counts()

eval["pred_roberta"] = pred

print(classification_report(eval["labels"],eval["pred_roberta"]))

print("final-f1 = ",f1_score(eval["labels"],eval["pred_roberta"]))

"""https://bmcgenomics.biomedcentral.com/articles/10.1186/s12864-019-6413-7"""

from sklearn.metrics import matthews_corrcoef
print("the  Matthews correlation coefficient (MCC) score is =", matthews_corrcoef(eval["labels"],eval["pred_roberta"]))

"""--------------
--------------

* **the final f1 score comes out to be 0.893**

* **the  Matthews correlation coefficient (MCC) score comes out to be 0.866**

--------------
--------------

# XLnet
"""

!pip install -U sentence-transformers

!pip install simpletransformers

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from simpletransformers.classification import (ClassificationModel, ClassificationArgs)
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

import torch

if torch.cuda.is_available():       
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

# Optional model configuration
model_args = ClassificationArgs(num_train_epochs=3)

# Create a ClassificationModel
model_3 = ClassificationModel("xlnet", "xlnet-base-cased")

from simpletransformers.classification import ClassificationModel, ClassificationArgs
model_args = ClassificationArgs(num_train_epochs=5,overwrite_output_dir=True,train_batch_size=16,max_seq_length = 128,output_dir ="/content/drive/MyDrive/TrueFoundry/XLNet_model")
# Create a ClassificationModel
model_3 = ClassificationModel(
     "xlnet", "xlnet-base-cased",
    num_labels=2,
    args=model_args)
#df_train = df_train[["text_filt","aspect","label"]]
#df_train.columns = ["text_a", "text_b", "labels"]
model_3.train_model(mydf)

pred,pred_prob = model_3.predict(eval[["text_a"]].values.tolist())

pd.Series(pred).value_counts()

eval["pred_XLNet"] = pred

print(classification_report(eval["labels"],eval["pred_XLNet"]))
print("final-f1 = ",f1_score(eval["labels"],eval["pred_XLNet"]))

from sklearn.metrics import matthews_corrcoef
print("the  Matthews correlation coefficient (MCC) score is =", matthews_corrcoef(eval["labels"],eval["pred_XLNet"]))

"""--------------
--------------

* **the final f1 score comes out to be 0.872**

* **the  Matthews correlation coefficient (MCC) score comes out to be 0.841**

--------------
--------------

# ELECTRA
"""

# Optional model configuration
model_args = ClassificationArgs(num_train_epochs=3)

# Create a ClassificationModel
model_4 = ClassificationModel("electra", "google/electra-base-discriminator")

from simpletransformers.classification import ClassificationModel, ClassificationArgs
model_args = ClassificationArgs(num_train_epochs=5,overwrite_output_dir=True,train_batch_size=16,max_seq_length = 128,output_dir ="/content/drive/MyDrive/TrueFoundry/ELECTRA_model")
# Create a ClassificationModel
model_4 = ClassificationModel(
     "electra", "google/electra-base-discriminator",
    num_labels=2,
    args=model_args)
#df_train = df_train[["text_filt","aspect","label"]]
#df_train.columns = ["text_a", "text_b", "labels"]
model_4.train_model(mydf)

pred,pred_prob = model_4.predict(eval[["text_a"]].values.tolist())

pd.Series(pred).value_counts()

eval["pred_ELECTRA"] = pred

print(classification_report(eval["labels"],eval["pred_ELECTRA"]))
print("final-f1 = ",f1_score(eval["labels"],eval["pred_ELECTRA"]))

from sklearn.metrics import matthews_corrcoef
print("the  Matthews correlation coefficient (MCC) score is =", matthews_corrcoef(eval["labels"],eval["pred_ELECTRA"]))

"""--------------
--------------

* **the final f1 score comes out to be 0.887**

* **the  Matthews correlation coefficient (MCC) score comes out to be 0.860**

--------------
--------------

# DistilroBERT
"""

# Optional model configuration
model_args = ClassificationArgs(num_train_epochs=3)

# Create a ClassificationModel
model_5 = ClassificationModel("roberta", "distilroberta-base")

from simpletransformers.classification import ClassificationModel, ClassificationArgs
model_args = ClassificationArgs(num_train_epochs=5,overwrite_output_dir=True,train_batch_size=16,max_seq_length = 128,output_dir ="/content/drive/MyDrive/TrueFoundry/DistilroBERT_model")
# Create a ClassificationModel
model_5 = ClassificationModel(
     "roberta", "distilroberta-base",
    num_labels=2,
    args=model_args)
#df_train = df_train[["text_filt","aspect","label"]]
#df_train.columns = ["text_a", "text_b", "labels"]
model_5.train_model(mydf)

pred,pred_prob = model_5.predict(eval[["text_a"]].values.tolist())

pd.Series(pred).value_counts()

eval["pred_DistilroBERT"] = pred

print(classification_report(eval["labels"],eval["pred_DistilroBERT"]))
print("final-f1 = ",f1_score(eval["labels"],eval["pred_DistilroBERT"]))

from sklearn.metrics import matthews_corrcoef
print("the  Matthews correlation coefficient (MCC) score is =", matthews_corrcoef(eval["labels"],eval["pred_DistilroBERT"]))

"""--------------
--------------

* **the final f1 score comes out to be 0.875**

* **the  Matthews correlation coefficient (MCC) score comes out to be 0.845**

--------------
--------------
"""

eval
