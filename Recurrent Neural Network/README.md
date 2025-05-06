# Spam Message Classification using RNNs (LSTM, Bi-LSTM, GRU)

Personal project on natural language processing (NLP) using recurrent neural networks (RNNs). In this notebook, I build and compare deep learning models to classify SMS messages as either *spam* or *ham* (not spam).

The main goal here is to practice building models with LSTM, Bi-LSTM, and GRU layers using TensorFlow/Keras, as well as to explore basic text preprocessing and evaluation techniques.

## Problem Overview

This project focuses on building RNN-based models for text classification. The goal is to develop models that can detect whether a given SMS message is *spam* or *ham* (not spam), which is a typical binary classification task in NLP.

I'll go through the following steps:
- Data preprocessing
- Tokenization and padding
- Building LSTM, Bi-LSTM, and GRU models
- Training and evaluation

## References
- Inspired by: [Text Classification using LSTM, Bi-LSTM, and GRU](https://nzlul.medium.com/the-classification-of-text-messages-using-lstm-bi-lstm-and-gru-f79b207f90ad)
- Keras documentation on RNN layers: https://keras.io/api/layers/recurrent_layers/
- TensorFlow RNN guide: https://www.tensorflow.org/guide/keras/working_with_rnns


## Dataset

SMS message dataset publicly available from UCL datasets (https://archive.ics.uci.edu/dataset/228/sms+spam+collection). It can also be downloaded from https://raw.githubusercontent.com/kenneth-lee-ch/SMS-Spam-Classification/master/spam.csv<br> The dataset contains 5,574 messages labeled as spam or not spam (ham).

The Pandas library was used to read and manipulate the dataset.

## Exploratory analysis

Most frequent words in the 'ham' class, according to the word cloud: now, will, ok, today, sorry, etc.

![image](https://github.com/user-attachments/assets/cc6ecaf6-f311-47f3-9f17-9c8deebeb405)


Most frequent words in the 'spam' class, according to the word cloud: FREE, call, URGENT, mobile, etc.

![image](https://github.com/user-attachments/assets/97368fc0-63ee-44d7-920b-f0e180f8e793)

Original distribution of classes.

![image](https://github.com/user-attachments/assets/de0feac1-0006-4f6e-b2ea-f2eb015d5cae)

Distribution after undersampling.

![image](https://github.com/user-attachments/assets/c66cea35-dd0d-484a-a151-6ee99c82dbc9)


## Model Configuration

### LSTM Bidirectional Single Layer

![image](https://github.com/user-attachments/assets/4ab13af3-ea34-40af-ad43-12ab6a0e20fe)

![image](https://github.com/user-attachments/assets/a2fbd410-5389-46dc-bcae-b98737c5ed6f)

### 3 Layers LSTM Bidirectional

![image](https://github.com/user-attachments/assets/f5692394-75b7-4af5-af35-e4d02af376e0)

![image](https://github.com/user-attachments/assets/578358a8-49b3-4662-bd38-bead192fadc5)

### Gated Recurrent Unit (GRU)

![image](https://github.com/user-attachments/assets/400d63e8-85ec-42c5-bd70-6eb78008bd42)

![image](https://github.com/user-attachments/assets/a2442a3c-5de0-430d-b6c2-ace375d49513)


## Results  

![image](https://github.com/user-attachments/assets/1a99d595-b5a7-4332-b370-2e338ccbf428)


The GRU model achieved the best accuracy, being 1.2% better than the Stacked LSTM model. The loss value for both GRU and Stacked LSTM are equivalent.
