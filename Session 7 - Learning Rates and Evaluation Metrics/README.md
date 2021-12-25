# **Evaluation Metrics**

### Precision:

Precision is a metric that quantifies the number of correct positive predictions made. It is calculated as the ratio of correctly predicted positive examples divided by the total number of positive examples that were predicted.

It measures the exactness of a classifier. A higher precision means less false positive, while a lower precision means more false positives.

Consider a spam classifier predicts 120 examples as spam, 90 of which are correct, and 30 of which are incorrect.

The precision for this model is calculated as:

- Precision = TruePositives / (TruePositives + FalsePositives)

- Precision = 90 / (90 + 30)

- Precision = 90 / 120

- Precision = 0.75
  



### Recall:

- Recall quantifies the number of positive class predictions made out of all positive examples in the dataset.

Consider a spam classifier makes predictions and predicts 90 of the positive class predictions correctly and 10 incorrectly. We can calculate the recall for this model as follows:

- Recall = TruePositives / (TruePositives + FalseNegatives)
- Recall = 90 / (90 + 10)
- Recall = 90 / 100
- Recall = 0.9

This model has a good recall.



### F1-Score:

- F-Measure provides a single score that balances both the concerns of precision and recall in one number.
- recall is calculated as the sum of true positives across all classes divided by the sum of true positives and false negatives across all classes. 
- A model predicts 77 examples correctly and 23 incorrectly for class 1, and 95 correctly and five incorrectly for class 2. We can calculate recall for this model as follows:
  - Recall = (TruePositives_1 + TruePositives_2) / ((TruePositives_1 + TruePositives_2) + (FalseNegatives_1 + FalseNegatives_2))
  - Recall = (77 + 95) / ((77 + 95) + (23 + 5))
  - Recall = 172 / (172 + 28)
  - Recall = 172 / 200
  - Recall = 0.86

## BLEU

BLEU, or the Bilingual Evaluation Understudy, is a score for comparing a candidate translation of text to one or more reference translations.

Although developed for translation, it can be used to evaluate text generated for a suite of natural language processing tasks.

Model BLEU Score value is 0.059

## Perplexity Metrics
Probability of a sentence can be defined as the product of the probability of each symbol given the previous symbols

Consider an arbitrary language . In this case, English will be utilized to simplify the arbitrary language. A language model assigns probabilities to sequences of arbitrary symbols such that the more likely a sequence  is to exist in that language, the higher the probability. A symbol can be a character, a word, or a sub-word (e.g. the word ‘going’ can be divided into two sub-words: ‘go’ and ‘ing’). Most language models estimate this probability as a product of each symbol's probability given its preceding symbols:

![alt text](https://raw.githubusercontent.com/thamizhannal/END3.0/main/Session%207%20-%20Learning%20Rates%20and%20Evaluation%20Metrics/imgs/perplexity.png?raw=true)


Word Unigram was applied to compute Perplexity. PPL = 4.541

## BERTScore
BERTScore leverages the pre-trained contextual embeddings from BERT and matches words in candidate and reference sentences by cosine similarity. 

BERT Score: Precision=13.80, Recall=28.83, F1-score=0.20



#### References:
https://huggingface.co/metrics/bertscore
https://colab.research.google.com/drive/1kpL8Y_AnUUiCxFjhxSrxCsc6-sDMNb_Q
https://thegradient.pub/understanding-evaluation-metrics-for-language-models/
