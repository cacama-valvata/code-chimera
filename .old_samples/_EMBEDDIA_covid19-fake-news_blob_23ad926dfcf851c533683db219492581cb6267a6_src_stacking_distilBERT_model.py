# -*- coding: utf-8 -*-
"""
Created on Fry October 23 2020

@author: Stepišnik Perdih
"""

import xml.etree.ElementTree as ET
import config
import numpy
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression, SGDClassifier
import parse_data
import time
import csv
import config
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import TruncatedSVD
from keras.preprocessing.sequence import pad_sequences
import pickle
import nltk
import torch
from transformers import DistilBertModel, DistilBertConfig, DistilBertForTokenClassification, DistilBertForSequenceClassification, DistilBertTokenizer
nltk.download('averaged_perceptron_tagger')
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig
import time
import datetime
import numpy as np
import random
from sklearn.metrics import f1_score
from numba import jit, cuda, vectorize
from transformers import get_linear_schedule_with_warmup
import pandas as pd

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def load_dataset():
    """
    Loads train and validation sets from .csv file
    """
    print("Loading dataset...")
    train_set = parse_data.get_train()
    valid_set = parse_data.get_dev()
    test_set = parse_data.get_test()
    print("Dataset loaded")

    return train_set, valid_set, test_set

def tokenize_dataset(text):
    """
        This method tokenizes text sentences and creates attention masks
    """
    print("Loading BERT tokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    token_ids = []
    for sentence in text['text_a']:
        if(len(sentence) > 512):
            sentence = sentence[:512]
        tokens = tokenizer.encode(sentence, add_special_tokens=True)
        token_ids.append(tokens)

    print('Max sentence length: ', max([len(sen) for sen in token_ids]))

    # Set the maximum sequence length.
    MAX_LEN = 512
    print('\nPadding/truncating all sentences to %d values...' % MAX_LEN)
    print('\nPadding token: "{:}", ID: {:}'.format(tokenizer.pad_token, tokenizer.pad_token_id))
    # Pad our input tokens with value 0.
    token_ids = pad_sequences(token_ids, maxlen=MAX_LEN, dtype="long",
                              value=0, truncating="pre", padding="post")
    print('\Done.')

    # Create attention masks
    attention_masks = []
    # For each sentence...
    for sent in token_ids:
        # Create the attention mask.
        #   - If a token ID is 0, then it's padding, set the mask to 0.
        #   - If a token ID is > 0, then it's a real token, set the mask to 1.
        att_mask = [int(token_id > 0) for token_id in sent]

        # Store the attention mask for this sentence.
        attention_masks.append(att_mask)


    return token_ids, attention_masks


def train(tokenized_sentences, mask, labels, validation_tokenized_sentences, validation_masks_parameter, validation_labels_parameter):
    """
    Trains BERT classifier
    """
    
    print(torch.version.cuda)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    print('Using device:', device)
    print()

    #Additional Info when using cuda
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')


    # Convert all inputs and labels into torch tensors, the required datatype
    # for our model.
    train_inputs = torch.tensor(tokenized_sentences)
    validation_inputs = torch.tensor(validation_tokenized_sentences)
    train_labels = torch.tensor(labels)
    validation_labels = torch.tensor(validation_labels_parameter)
    train_masks = torch.tensor(mask)
    validation_masks = torch.tensor(validation_masks_parameter)


    # The DataLoader needs to know our batch size for training, so we specify it
    # here.
    # For fine-tuning BERT on a specific task, the authors recommend a batch size of
    # 16 or 32.
    batch_size = 32
    # Create the DataLoader for our training set.
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    # Create the DataLoader for our validation set.
    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)


    # Load BertForSequenceClassification, the pretrained BERT model with a single
    # linear classification layer on top.
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
        num_labels = 2, # The number of output labels--2 for binary classification.
                        # You can increase this for multi-class tasks.
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False, # Whether the model returns all hidden-states.
    )
    model.cuda()


    # Get all of the model's parameters as a list of tuples.
    params = list(model.named_parameters())
    print('The BERT model has {:} different named parameters.\n'.format(len(params)))
    print('==== Embedding Layer ====\n')
    for p in params[0:5]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
    print('\n==== First Transformer ====\n')
    for p in params[5:21]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
    print('\n==== Output Layer ====\n')
    for p in params[-4:]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))


    # Note: AdamW is a class from the huggingface library (as opposed to pytorch)
    # I believe the 'W' stands for 'Weight Decay fix"
    optimizer = AdamW(model.parameters(),
                      lr=2e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                      eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                      )

    # Number of training epochs (authors recommend between 2 and 4)
    epochs = 2
    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * epochs
    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)

    # This training code is based on the `run_glue.py` script here:
    # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128
    # Set the seed value all over the place to make this reproducible.
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    # Store the average loss after each epoch so we can plot them.
    loss_values = []
    # For each epoch...
    for epoch_i in range(0, epochs):

        # ========================================
        #               Training
        # ========================================


        # Perform one full pass over the training set.
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')
        # Measure how long the training epoch takes.
        t0 = time.time()
        # Reset the total loss for this epoch.
        total_loss = 0
        # Put the model into training mode. Don't be mislead--the call to
        # `train` just changes the *mode*, it doesn't *perform* the training.
        # `dropout` and `batchnorm` layers behave differently during training
        # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        model.train()
        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)

                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
            # Unpack this training batch from our dataloader.
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using the
            # `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids
            #   [1]: attention masks
            #   [2]: labels
            b_input_ids = batch[0]
            b_input_ids = b_input_ids.type(torch.LongTensor)
            b_input_ids = b_input_ids.to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            # Always clear any previously calculated gradients before performing a
            # backward pass. PyTorch doesn't do this automatically because
            # accumulating the gradients is "convenient while training RNNs".
            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            model.zero_grad()
            # Perform a forward pass (evaluate the model on this training batch).
            # This will return the loss (rather than the model output) because we
            # have provided the `labels`.
            # The documentation for this `model` function is here:
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification

            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask,
                            labels=b_labels)

            # The call to `model` always returns a tuple, so we need to pull the
            # loss value out of the tuple.
            loss = outputs[0]
            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value
            # from the tensor.
            total_loss += loss.item()
            # Perform a backward pass to calculate the gradients.
            loss.backward()
            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()
            # Update the learning rate.
            scheduler.step()
        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)

        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)
        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))

        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.
        print("")
        print("Running Validation...")
        t0 = time.time()
        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        model.eval()
        # Tracking variables
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        # Evaluate data for one epoch
        preds = []
        orgs = []
        for batch in validation_dataloader:
            # Add batch to GPU
            """
            batch = tuple(t.to(device) for t in batch)

            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            """
            
            b_input_ids=batch[0].to(device)
            b_input_mask=batch[1].to(device)
            b_labels=batch[2].to(device)
   
            # Telling the model not to compute or store gradients, saving memory and
            # speeding up validation
            with torch.no_grad():
                # Forward pass, calculate logit predictions.
                # This will return the logits rather than the loss because we have
                # not provided labels.
                # token_type_ids is the same as the "segment ids", which
                # differentiates sentence 1 and 2 in 2-sentence tasks.
                # The documentation for this `model` function is here:
                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                outputs = model(b_input_ids,
                                token_type_ids=None,
                                attention_mask=b_input_mask)

            # Get the "logits" output by the model. The "logits" are the output
            # values prior to applying an activation function like the softmax.
            logits = outputs[0]
            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            guessed = np.argmax(logits,axis=1)
            print(guessed)
            label_ids = b_labels.to('cpu').numpy()
            preds = preds + guessed.tolist()
            orgs = orgs + label_ids.tolist()
            # Track the number of batches
            nb_eval_steps += 1
        print(f1_score(preds, orgs))
        print(label_ids)
        print("  Validation took: {:}".format(format_time(time.time() - t0)))
    print("")
    print("Training complete!")

    ## save model with pickle
    with open(os.path.join("clf_en.pkl"), mode='wb') as f:
        pickle.dump(model, f)


def convert_labels_to_ids(labels):
    """
    Converts labels to integer IDs
    """
    label_to_id = {}
    counter = 0
    id_list = []

    for label in labels:
        if label not in label_to_id.keys():
            label_to_id[label] = counter
            counter += 1
        id_list.append(label_to_id[label])
    return id_list

def fit_space(X, model_path="."):
    dictionary = {}
    dictionary['text_a'] = X
    feature_matrix, masks = tokenize_dataset(dictionary)
    return feature_matrix, masks


def fit(X, model_path="."):
    matrix, masks = fit_space(X, model_path)
    clf = import_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X = torch.tensor(matrix)
    masks = torch.tensor(masks)

    validation_data = TensorDataset(X, masks)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=8)

    clf.eval()
    # Tracking variables
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    # Evaluate data for one epoch
    preds = []
    orgs = []

    for batch in validation_dataloader:
        b_input_ids = batch[0]
        b_input_ids = b_input_ids.type(torch.LongTensor)
        b_input_ids = b_input_ids.to(device)
        b_input_mask = batch[1].to(device)
            
        # Telling the model not to compute or store gradients, saving memory and
        # speeding up validation
        with torch.no_grad():
            outputs = clf(b_input_ids,
                          token_type_ids=None,
                          attention_mask=b_input_mask)

        # Get the "logits" output by the model. The "logits" are the output
        # values prior to applying an activation function like the softmax.
        logits = outputs[0]
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        guessed = np.argmax(logits, axis=1)
        preds = preds + guessed.tolist()
        # Track the number of batches
        nb_eval_steps += 1
    return preds

def import_model(lang='en',path_in="."):
    clf = pickle.load(open(os.path.join(config.PICKLES_PATH, "clf_" + lang + ".pkl"), 'rb'))
    return clf

def evaluate(clf, X, y, masks):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X = torch.tensor(X)
    y = torch.tensor(y)
    masks = torch.tensor(masks)


    validation_data = TensorDataset(X, masks, y)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=8)

    clf.eval()
    # Tracking variables
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    # Evaluate data for one epoch
    preds = []
    orgs = []

    for batch in validation_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        # Telling the model not to compute or store gradients, saving memory and
        # speeding up validation
        with torch.no_grad():
            outputs = clf(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask)

        # Get the "logits" output by the model. The "logits" are the output
        # values prior to applying an activation function like the softmax.
        logits = outputs[0]
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        guessed = np.argmax(logits, axis=1)
        print(guessed)
        label_ids = b_labels.to('cpu').numpy()
        preds = preds + guessed.tolist()
        orgs = orgs + label_ids.tolist()
        # Track the number of batches
        nb_eval_steps += 1
    print(f1_score(preds, orgs))
    print(label_ids)


if __name__ == "__main__":
    train_set, valid_set, test_set = load_dataset()
    #tokenized_sentences, attention_mask = tokenize_dataset(train_set)
    #validation_tokenized_sentences, validation_masks = tokenize_dataset(valid_set)
    #test_tokenized_sentences, test_masks = tokenize_dataset(test_set)
    #labels = convert_labels_to_ids(train_set['label'])
    #validation_labels = convert_labels_to_ids(valid_set['label'])
    #test_labels = convert_labels_to_ids(test_set['label'])

    #pd.DataFrame(tokenized_sentences).to_csv("train_features.csv")
    #pd.DataFrame(attention_mask).to_csv("train_masks.csv")

    #pd.DataFrame(validation_tokenized_sentences).to_csv("validation_features.csv")
    #pd.DataFrame(validation_masks).to_csv("validation_masks.csv")

    #pd.DataFrame(test_tokenized_sentences).to_csv("test_features.csv")
    #pd.DataFrame(test_masks).to_csv("test_masks.csv")
    #train(tokenized_sentences, attention_mask, labels, validation_tokenized_sentences, validation_masks, validation_labels)

    train_text = [
        "Brexit (/ˈbrɛksɪt, ˈbrɛɡzɪt/;[1] a portmanteau of British and exit) is the withdrawal of the United Kingdom (UK) from the European Union (EU). Following a referendum held on 23 June 2016 in which 51.9 per cent of those voting supported leaving the EU, the Government invoked Article 50 of the Treaty on European Union, starting a two-year process which was due to conclude with the UK's exit on 29 March 2019 – a deadline which has since been extended to 31 October 2019.[2]",
        "Withdrawal from the EU has been advocated by both left-wing and right-wing Eurosceptics, while pro-Europeanists, who also span the political spectrum, have advocated continued membership and maintaining the customs union and single market. The UK joined the European Communities (EC) in 1973 under the Conservative government of Edward Heath, with continued membership endorsed by a referendum in 1975. In the 1970s and 1980s, withdrawal from the EC was advocated mainly by the political left, with the Labour Party's 1983 election manifesto advocating full withdrawal. From the 1990s, opposition to further European integration came mainly from the right, and divisions within the Conservative Party led to rebellion over the Maastricht Treaty in 1992. The growth of the UK Independence Party (UKIP) in the early 2010s and the influence of the cross-party People's Pledge campaign have been described as influential in bringing about a referendum. The Conservative Prime Minister, David Cameron, pledged during the campaign for the 2015 general election to hold a new referendum—a promise which he fulfilled in 2016 following pressure from the Eurosceptic wing of his party. Cameron, who had campaigned to remain, resigned after the result and was succeeded by Theresa May, his former Home Secretary. She called a snap general election less than a year later but lost her overall majority. Her minority government is supported in key votes by the Democratic Unionist Party.",
        "The broad consensus among economists is that Brexit will likely reduce the UK's real per capita income in the medium term and long term, and that the Brexit referendum itself damaged the economy.[a] Studies on effects since the referendum show a reduction in GDP, trade and investment, as well as household losses from increased inflation. Brexit is likely to reduce immigration from European Economic Area (EEA) countries to the UK, and poses challenges for UK higher education and academic research. As of May 2019, the size of the divorce bill—the UK's inheritance of existing EU trade agreements—and relations with Ireland and other EU member states remains uncertain. The precise impact on the UK depends on whether the process will be a hard or soft Brexit."]

    print(fit(train_text))



    #clf = import_model()
    #evaluate(clf, test_tokenized_sentences, test_labels, test_masks)
