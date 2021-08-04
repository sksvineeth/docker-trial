#!/usr/bin/env python
# coding: utf-8

# ## Functional API Approach for Customs Codes
# Goal: Include catergorical data as input to the model and concat with previous LSTM layer

# ### The goal of this approach is to take multiple type of inputs - namely categorical and non-categorical data (text fields) for the prediction of Customs Codes
# 
# * Categorical features used: - Brand & Dm_Class
# * Non-categorical features: - combination of Medium_description and GDP Description

# In[2]:


#installing necessary packages
#!pip install pydot
#!pip install pydotplus
#!pip install graphviz
get_ipython().system('pip install bayesian-optimization')


# In[3]:


#importing necessary libraries
import pandas as pd
import numpy as np
from numpy import unique
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Flatten, LSTM, Dropout, Activation, Embedding, Bidirectional, Input
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import multi_gpu_model, plot_model
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs
import os
import time
import io
import boto3
import pickle
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix, plot_precision_recall_curve, plot_roc_curve
import traceback



# In[4]:



# ### 1 . Load & Prepare Data

# In[5]:


def data_process():

    print("Imported!")
    bucket = "innovation-poc"
    train_key = "product_classifier_data_correlation_Brigitte_data_owner/NL & US Data/features_preprocessed.csv"
    #Load Training Data
    buffer = io.BytesIO()
    s3 = boto3.resource('s3')
    s3_object = s3.Object(bucket, train_key)
    s3_object.download_fileobj(buffer)
    print(s3_object)
    print(buffer.seek(0))
    df = pd.read_csv(buffer)
    df = df.loc[df['Assurance Level'].isin(['2.0','3.0', '4.0'])]
    df = df.reset_index(drop = True)
    print(df.shape)
    
    target_field = "Target"
    y = df[target_field]
    
    df["Medium_Description"].fillna("<missing>", inplace=True)
    df["GDP_y"].fillna("<missing>", inplace=True)
    df["combined_desc"] = df.apply(lambda row: str(row["Medium_Description"]) + " " + str(row["GDP_y"]),
                                axis=1)
    
    training_idx = df.loc[(df['Used_for']=='Training')].index.values.tolist()
    test_idx =  df.loc[df['Used_for'] != 'Training'].index.values.tolist()
    
    
    #test train split 
    y_train, y_test = y[training_idx], y[test_idx] 
    X_train, X_test = df.iloc[training_idx], df.iloc[test_idx]
    
    lables = pd.get_dummies(y_train).columns
    y_train = pd.get_dummies(y_train).values
    y_test = pd.get_dummies(y_test)
    y_test = y_test.reindex(columns=lables).fillna(0) 
    print(y_train.shape)
    return X_train, y_train, X_test, y_test, training_idx, test_idx


# In[6]:

# ### 2. Preprocess the data

# In[7]:


def preprocess(X_train, y_train, X_test, y_test):
    #lables = pd.get_dummies(y_train).columns
    #y_train = pd.get_dummies(y_train).values
    #y_test = pd.get_dummies(y_test)
    #y_test = y_test.reindex(columns=lables).fillna(0) 
    
    vocab_size = 5000 # make the top list of words (common words)
    embedding_dim = 64
    max_length = 100 #max length of each word in vocab
    trunc_type = 'post' 
    padding_type = 'post'
    oov_tok = '<OOV>' # OOV = Out of Vocabulary
    
    tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(X_train['combined_desc'])
    word_index = tokenizer.word_index
    #print(f'Found {len(word_index)} unique words.')
    
    #for train and test set 
    X_train_desc = tokenizer.texts_to_sequences(X_train['combined_desc'])
    X_train_desc = pad_sequences(X_train_desc, maxlen=max_length)
    X_test_desc = tokenizer.texts_to_sequences(X_test['combined_desc'])
    X_test_desc = pad_sequences(X_test_desc, maxlen=max_length)
    
    #load categorical data and prepare inputs
    feature_list = [
    "Brand",
    "Dm_Class"
    ]
    train_features = X_train[feature_list]
    test_features = X_test[feature_list]
    train_features.fillna("<missing>", inplace=True)
    test_features.fillna("<missing>", inplace=True)
    return train_features, test_features, X_train_desc


# In[8]:

# ### 3 .  Prepare Inputs

# In[9]:


# prepare input data
def prepare_inputs(X_train, X_test):
    X_train_enc_list, X_test_enc_list, X_cat_size = list(), list(), list()
    # label encode each column
    for i in range(X_train.shape[1]):
        le = LabelEncoder()
        le.fit(X_train.iloc[:, i])
        X_test.iloc[:, i] = X_test.iloc[:, i].map(lambda s: '<unknown>' if s not in le.classes_ else s)
        le.classes_ = np.append(le.classes_, '<unknown>')
        X_cat_size.append(len(le.classes_))
        # encode
        train_enc = le.transform(X_train.iloc[:, i])
        test_enc = le.transform(X_test.iloc[:, i])
        # store
        X_train_enc_list.append(train_enc)
        X_test_enc_list.append(test_enc)
    X_train_enc = pd.DataFrame(X_train_enc_list)
    X_test_enc = pd.DataFrame(X_test_enc_list)
    return X_train_enc_list, X_test_enc_list, X_cat_size

 ### 4.  Split the data and Build the Model

# split training data into training and validation set
def val_split(train_cat, train_desc, y_train):
    val_size = 0.2
    X_train_cat, X_val_cat = list(), list()
    indices = np.random.permutation(y_train.shape[0]) #mix up all of the testing indices
    train_max_idx = int(y_train.shape[0] * (1-val_size)) #set the size of the training group
    training_idx, val_idx = indices[:train_max_idx], indices[train_max_idx:]
    for i in range(len(train_cat)):
        row = train_cat[i]
        X_train_cat.append(row[training_idx])
        X_val_cat.append(row[val_idx])
    X_train_desc, X_val_desc = train_desc[training_idx,:], train_desc[val_idx,:]
    y_train, y_val = y_train[training_idx], y_train[val_idx]
    #y_train, y_val = y_train.str.slice(start = training_idx), y_train.str.slice(start = val_idx)

    return np.array(X_train_cat), np.array(X_val_cat), X_train_desc, X_val_desc, y_train, y_val


# ####  Build the model architecture and combine LSTM and Categorical layers


def build_model(lstm_nodes, dense_nodes, dropout=.2):
    in_layers = list()
    em_layers = list()
    
    vocab_size = 5000 # make the top list of words (common words)
    embedding_dim = 64
    max_length = 100 #max length of each word in vocab
    trunc_type = 'post' 
    padding_type = 'post'
    oov_tok = '<OOV>' # OOV = Out of Vocabulary
        #create layers for categorical embeddings
    for i in range(len(X_train_cat)):
        # calculate the number of unique inputs
        n_labels = X_cat_size[i]
        # define input layer
        in_layer = Input(shape=(1,))
        # define embedding layer
        em_layer = Embedding(n_labels, 10)(in_layer)
        # store layers
        in_layers.append(in_layer)
        em_layers.append(em_layer)
    cat_merge = layers.concatenate(em_layers)
    cat_dense = Dense(dense_nodes, activation='relu')(cat_merge)
    cat_flat = Flatten()(cat_dense)
    #create layer for LSTM and embedding
    desc_input = Input(shape=(None,), name='desc') #variable input length
    in_layers.append(desc_input)
    desc_input = Embedding(vocab_size, embedding_dim)(desc_input)
    desc_lstm = LSTM(lstm_nodes, dropout=dropout)(desc_input)
        
    feature_merge = layers.concatenate([desc_lstm, cat_flat])
    
    output = Dense(y_train.shape[1],activation='softmax')(feature_merge)
    model = Model(inputs = in_layers, outputs = output,)
    return model



def train():   
    print('Compiling and building the model...')
    try:
        model = build_model(95, 73, .32)
        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy', keras.metrics.Recall(), keras.metrics.Precision()],
            )
        tf.keras.backend.clear_session()
        # split training and validation set using val-split function
        #X_train_cat, X_val_cat, X_train_desc, X_val_desc, y_train, y_val = val_split(X_train_cat, X_train_desc, y_train)
        monitor = EarlyStopping(monitor='val_accuracy', patience=3, min_delta=0.01,
                            mode='auto', restore_best_weights=True)
        lstm = model.fit([X_train_cat[0], X_train_cat[1], X_train_desc], y_train, 
                  validation_data = ([X_val_cat[0], X_val_cat[1], X_val_desc], y_val), 
                  callbacks=[monitor], epochs = 20, batch_size = 32, verbose = 1)
         # save the model
        pkl_path = "product_classifier_data_correlation_Brigitte_data_owner/NL & US Data/"

        s = pickle.dumps(lstm)
        s3_resource = boto3.resource('s3')
        s3_resource.Object(bucket,pkl_path).put(Body=s)
        print('Training complete. Model saved to s3')
    except Exception as e:
        # Write out an error file. This will be returned as the failureReason in the
        # DescribeTrainingJob result.
        trc = traceback.format_exc()
        s = pickle.dumps(trc)
        s3_resource = boto3.resource('s3')
        s3_resource.Object(bucket,pkl_path).put(Body=s)
            # Printing this causes the exception to be in the training job logs, as well.
        print('Exception during training: ' + str(e) + '\n' + trc, file=sys.stderr)
            # A non-zero exit code causes the training job to be marked as Failed.
        sys.exit(255)


if __name__ == '__main__':
    X_train, y_train, X_test, y_test, training_idx, test_idx = data_process()
    train_features, test_features, X_train_desc = preprocess(X_train, y_train, X_test, y_test)
    X_train_cat, X_test_cat, X_cat_size = prepare_inputs(train_features, test_features)
    X_train_cat, X_val_cat, X_train_desc, X_val_desc, y_train, y_val = val_split(X_train_cat, X_train_desc, y_train)
    model = build_model(95, 73, .32)
    train()
    # A zero exit code causes the job to be marked a Succeeded.
    sys.exit(0)

