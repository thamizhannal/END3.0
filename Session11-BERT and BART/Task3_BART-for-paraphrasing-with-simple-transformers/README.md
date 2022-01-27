# BART for Paraphrasing with Simple Transformers
### Learning Objective: 
Reproduce the training explained in this blog (https://towardsdatascience.com/bart-for-paraphrasing-with-simple-transformers-7c9ea3dfdd8c.). You can decide to pick fewer datasets.

### BART description 
BART is a denoising autoencoder for pretraining sequence-to-sequence models. It is trained by (1) corrupting text with an arbitrary noising function, and (2) learning a model to reconstruct the original text. It uses a standard Transformer-based neural machine translation architecture. It uses a standard seq2seq/NMT architecture with a bidirectional encoder (like BERT) and a left-to-right decoder (like GPT). This means the encoder's attention mask is fully visible, like BERT, and the decoder's attention mask is causal, like GPT2.

![BART Arch](https://production-media.paperswithcode.com/methods/Screen_Shot_2020-06-01_at_9.49.47_PM.png)

The Bidirectional and Auto-Regressive Transformer or BART is a Transformer that combines the Bidirectional Encoder (i.e. BERT like) with an Autoregressive decoder (i.e. GPT like) into one Seq2Seq model. In other words, it gets back to the original Transformer architecture proposed by Vaswani, albeit with a few changes.

Recall BERT, which has a Bidirectional Transformer with a Masked Language Modelling (and NSP) prediction task — where the goal is to predict the missing samples:

![BART](https://github.com/thamizhannal/END3.0/blob/main/Session11-BERT%20and%20BART/Task3_BART-for-paraphrasing-with-simple-transformers/imgs/BERT.png)

Also recall GPT, where the model is autoregressive, and where the task is to predict the next token:
![GPT](https://raw.githubusercontent.com/thamizhannal/END3.0/main/Session11-BERT%20and%20BART/Task3_BART-for-paraphrasing-with-simple-transformers/imgs/decoder.png)
 

BART combines the two approaches into one:
![BART](https://raw.githubusercontent.com/thamizhannal/END3.0/main/Session11-BERT%20and%20BART/Task3_BART-for-paraphrasing-with-simple-transformers/imgs/BART.png)
 
### Training log snippets
### 5 sample results
