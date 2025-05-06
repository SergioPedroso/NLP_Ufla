## Problem Statement

Develop an NLP model for text classification based on BERT.

The task involves classifying Brazilian‑Portuguese texts from the **HateBR** dataset:

1. **Binary classification** – detect whether a text contains offensive language.
2. **Multi‑class classification** – determine the offensiveness level (slight, moderate, severe).

### Key Steps

* Load the dataset with the **🤗 Datasets** library.
* Pre‑process text using a BERT tokenizer.
* Fine‑tune **BERTimbau** on HateBR data.
* Evaluate the fine‑tuned model and run inference.

### References

* [https://huggingface.co/docs/transformers/tasks/sequence\_classification](https://huggingface.co/docs/transformers/tasks/sequence_classification)
* [https://huggingface.co/course/chapter1/1](https://huggingface.co/course/chapter1/1)
