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

### References
- Inspired by: [Text Classification using LSTM, Bi-LSTM, and GRU](https://nzlul.medium.com/the-classification-of-text-messages-using-lstm-bi-lstm-and-gru-f79b207f90ad)
- Keras documentation on RNN layers: https://keras.io/api/layers/recurrent_layers/
- TensorFlow RNN guide: https://www.tensorflow.org/guide/keras/working_with_rnns


### Dataset

Base de dados de mensagens SMS de celulares, publicamente disponível na UCL datasets (https://archive.ics.uci.edu/dataset/228/sms+spam+collection). Pode também ser baixada de https://raw.githubusercontent.com/kenneth-lee-ch/SMS-Spam-Classification/master/spam.csv<br>
O dataset contém 5.574 mensagens rotuladas como *spam* ou não *spam* (*ham*).

A biblioteca Pandas foi usada para ler e manipular o dataset.


![image](https://github.com/user-attachments/assets/cc6ecaf6-f311-47f3-9f17-9c8deebeb405)
