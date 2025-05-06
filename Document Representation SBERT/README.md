# Semantic Search and Text Classification with SentenceBERT

## Objective

The goal of this project is to implement a semantic search application and a text classifier, both utilizing document embeddings obtained from the SentenceBERT (SBERT) model. This model is employed to convert input texts into numerical representations known as "embeddings" (n-dimensional numerical vectors).

This project will help practice the following skills:
- Generating text embeddings using the SentenceBERT model.
- Using Cosine Similarity to measure similarity between texts.
- Classifying e-commerce texts into specific categories.

References:
- [SBERT.net](https://www.sbert.net/)
- [An Intuitive Explanation of Sentence-BERT (Towards Data Science)](https://towardsdatascience.com/an-intuitive-explanation-of-sentence-bert-1984d144a868)
- [The Rise of Sentence BERT: A Game Changer for Semantic Search (Medium)](https://medium.com/@gulsum.budakoglu/the-rise-of-sentence-bert-a-game-changer-for-semantic-search-1a857c1923aa)
- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks (ACLanathology)](https://aclanthology.org/D19-1410.pdf)
- [Semantic Search using Sequence-BERT (Medium)](https://medium.com/@jeremyarancio/semantic-search-using-sequence-bert-2116dabecfa3)
- [SBERT.net Semantic Search Utility](https://www.sbert.net/examples/applications/semantic-search/README.html#util-semantic-search)
- [Colab Example](https://colab.research.google.com/drive/12cn5Oo0v3HfQQ8Tv6-ukgxXSmT3zl35A?usp=sharing)

## Generating Embeddings with the SentenceBERT Model

Transformer-based models like BERT and RoBERTa address various NLP tasks by computing word-level embeddings. However, for tasks such as semantic search, which require a robust sentence-level understanding, using word-level Transformers becomes computationally prohibitive.

Semantic search involves finding sentences (or documents composed of multiple sentences) that are semantically similar to a given target sentence/document. Comparing sentence pairs using traditional Transformers for a large corpus is very time-consuming.

SentenceBERT (SBERT) is a modification of the standard pre-trained BERT network. It uses Siamese Networks and a triplet loss function to create sentence embeddings. These embeddings can then be compared using Cosine Similarity or other similarity/distance metrics, making semantic search for large numbers of sentences highly efficient.

SentenceBERT modifies the BERT model by adding a pooling operation to its output, resulting in a fixed-size sentence embedding.

To train a Siamese Network, pairs of positive (semantically similar) and negative (semantically dissimilar) sentences are used. For example, consider Sentence A and Sentence B, which are semantically similar; this pair is a positive example. Another pair, Sentence A and Sentence C, which are not similar, forms a negative example. The network's training objective is the Triplet Loss function, which aims to minimize the Euclidean distance between Sentence A and Sentence B (similar) and maximize the distance between Sentence A and Sentence C (dissimilar).

A Siamese Network consists of two identical subnetworks that share weights. Parameter updates are mirrored across both subnetworks. After training, a new sentence can be encoded by passing it through one of the subnetworks, and its embedding is obtained from the pooling layer. This embedding can then be used for tasks like search and classification, serving as a feature vector for machine learning models.

The primary difference between BERT and SentenceBERT models lies in their encoder structure. BERT uses a Cross-Encoder, while SentenceBERT uses a Bi-Encoder.

- **Bi-Encoder:** Sentence embeddings (u and v) are created separately for each sentence. These embeddings are then compared using a similarity function like Cosine Similarity to obtain a similarity score.
- **Cross-Encoder:** Both sentences are passed into the Transformer network simultaneously. The output is a similarity score, typically between 0 and 1. The Cross-Encoder does not produce individual sentence embeddings.

More details can be found in the paper: [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://aclanthology.org/D19-1410.pdf)

## Cosine Similarity

Cosine Similarity measures the similarity between two vectors A and B by the cosine of the angle between them. It utilizes the concepts of the Dot Product and Vector Norm.

**Dot Product** <br>
Consider two vectors A and B: <br>
$A = (a_1, a_2, \dots, a_n)$ <br>
$B = (b_1, b_2, \dots, b_n)$ <br>
The Dot Product of A and B is given by: <br>
$ A \cdot B = (a_1 \times b_1) + (a_2 \times b_2) + \dots + (a_n \times b_n) $ <br>

**Vector Norm** <br>
For a vector V: <br>
$V = (v_1, v_2, \dots, v_n)$ <br>
The Norm of V is given by: <br>
$||V|| = \sqrt{(v_1)^2 + (v_2)^2 + \dots + (v_n)^2}$

**Cosine Similarity** <br>
The Cosine Similarity between two vectors A and B is given by: <br>
$sim(A,B) = \cos(\theta) = \frac{A \cdot B} {||A|| \times ||B||}$

**Properties** <br>
- Value ranges between -1 and 1 (in general).
- Value ranges between 0 and 1 (in many natural language processing applications, especially with non-negative embeddings).
- $\leq 0$ indicates no similarity, and 1 indicates total similarity (in the context of 0-1 scaled similarity).

## Semantic Search

Given a query, semantic search aims to find the document most similar to that query or to generate a ranked list of documents in descending order of their similarity to the query.

A document can be a single sentence or a long text, such as a webpage, a scientific article, or a book.
In the case of SentenceBERT, the input text length is limited by the maximum input size of the underlying BERT model, which is typically 512 tokens. In terms of word count, this is effectively smaller, as individual words can be tokenized into multiple sub-word tokens.