# Text-Recognition-Using-Seq-to-Seq-Model
## Introduction
- In this repo, I will propose a seq-to-seq approach for Text Recognition (TR), this is also my challenge in my new comapny.
- This approach is a trial model for Text Recognition. Because I want to make a new installation of **Transformer** for TR, although I can retry a available paper. Thus, the first version maybe have a medium accuracy which is lower than many other models, but **WORTH A TRY**
## Algorithm
- In this section, I will explain my idea for this problem. Recently, many projects and papers propose several of new model for OCR problem, such as seq-to-seq, object recognition, ... My method also bases on seq-to-seq model, the new point in this repo is replacing RNN and Attention by Transformer whose original paper is [here](https://arxiv.org/pdf/1706.03762.pdf).
- My model seq-to-seq often contains Encoder and Decoder. Moreover, there is a CNN driven by Transformer to extract sequential feature as input for Transformer. And the key point of this system is **Multi-Head Attention**, so I will overal view attention mechanism of Transformer.
### 1. Transformer
<p align="center">
  <img src="/images/trans_0.png" width="350" />
  <img src="/images/trans_1.png" width="450" /> 
</p>

- Above pictures are illustrations of Transformer. You can see the way Attention is applied.
- Multi-Head Attention is used in seq-to-seq model to determine how much each word will be expressed at this position.
- Detail of components' size in Transformer:
  * Tokens: Ouput tokens are index vectors, and there is a Embedder to convert tokens to vector with size of d_model.
  * Encoder input: feature matrix from CNN and its size of [batch size, max_seq_len, d_model], max_seq_len is maximum number of words in a sentence, d_model is size of embeddings
  * Decoder input is target embedding: [batch_size, max_seq_len, d_model]

### 2. CNN
The network used in this repo is VGG16
## Dataset
[IAM dataset](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database)
## Reference
There are many papers about OCR problem in this [git](https://github.com/ChanChiChoi/awesome-ocr)
