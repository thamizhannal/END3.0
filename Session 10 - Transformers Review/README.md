# Session 10 - Transformers Review

## Assignment

Train the same code, but on different data. If you have n-classes, your accuracy MUST be more than 4 * 100 / n.

Submit the Github link, that includes your notebook with training logs, and proper readme file.



### Data Set: IWSLT2016: International Workshop on Spoken Language Translation (IWSLT)

This is a machine translation dataset that is focused on the automatic transcription and translation of TED and TEDx talks, i.e. public speeches covering many different topics. Compared with the WMT dataset, mentioned below, this dataset is relatively small (the corpus has 130K sentences) and therefore models should be able to achieve decent BLEU scores fast (in several hours).

Ref: http://www.cs.toronto.edu/~pekhimenko/tbd/datasets.html



#### Transformer Architecture:





![img](https://mchromiak.github.io/articles/2017/Sep/12/Transformer-Attention-is-all-you-need/img/encoder.png)

#### Parameters:

BATCH_SIZE = 64

INPUT_DIM = len(vocab_transform[SRC_LANGUAGE]) 

OUTPUT_DIM = len(vocab_transform[TGT_LANGUAGE])

HID_DIM = 256

ENC_LAYERS = 3

DEC_LAYERS = 3

ENC_HEADS = 8

DEC_HEADS = 8

ENC_PF_DIM = 512

DEC_PF_DIM = 512

ENC_DROPOUT = 0.1

DEC_DROPOUT = 0.1

### Embedding layer and Position Encoding

![img](https://mchromiak.github.io/articles/2017/Sep/12/Transformer-Attention-is-all-you-need/img/encoder.png)



### Self Attention:

![img](https://jalammar.github.io/images/t/transformer_self-attention_visualization.png)

### 

![img](https://jalammar.github.io/images/t/self-attention-output.png)



### Encoder-Decoder

Sub layer in encoder and decoder module.

![img](https://jalammar.github.io/images/t/transformer_resideual_layer_norm_3.png)



### Decoder:

![img](https://jalammar.github.io/images/t/transformer_decoding_2.gif)







### Model Training:

![image-20220122221949897](C:\Users\tparamas\AppData\Roaming\Typora\typora-user-images\image-20220122221949897.png)



### Conclusion:

The Seq2Seq transformer network was trained using IWSLT2016 dataset for 10 epoch's, from validation results metrics it is evident that validation PPL value reducing from 53 to 10 at 10th epoch and also validation loss reduced from 3.9 to 2.9 which shows it is good English to French translation model.