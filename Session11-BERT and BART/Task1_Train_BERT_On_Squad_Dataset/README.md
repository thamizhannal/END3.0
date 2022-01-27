# Train BERT Model on Squad DataSet  <br>
### Learning Objective: 
#### To Train BERT using the code mentioned here (https://drive.google.com/file/d/1Zp2_Uka8oGDYsSe5ELk-xz6wIX8OIkB7/view?usp=sharing) on the Squad Dataset for 20% overall samples (1/5 Epochs).
#### Show results on 5 samples.  <br>

### What is BERT ? <br>
BERT is a Bidirectional Transformer (basically transformer only) with a Masked Language Modelling (Links to an external site.) and Next Sentence Prediction (Links to an external site.) task, where the goal is to predict the missing samples. So Given A_C_E, predict B and D. BERT is an encode-only model.

BERT makes use of Transformer architecture (attention mechanism) that learns contextual relations between words in a text. In its vanilla form, Transformer includes two separate mechanisms - an encoder that reads the text input and a decoder that produces a prediction for the task

BERT falls into a self-supervised model category. That means, it can generate inputs and outputs from the raw corpus without being explicitly programmed by humans.

Since BERT's goal is to generate a language model, only the encoder mechanism is necessary.

As opposed to directional models, which read the text input sequentially (left to right or right to left), the Transformer encoder reads the entire sequence of words at once. 

Therefore it is considered bidirectional. This characteristic allows the model to learn the context of a word based on all of its surroundings (left and right of the word).

![high-level Transformer encoder](https://miro.medium.com/max/875/0*ViwaI3Vvbnd-CJSQ.png)

### What is SQuAD?
Stanford Question Answering Dataset (SQuAD) is a reading comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage, or the question might be unanswerable.
Ref: https://rajpurkar.github.io/SQuAD-explorer/

### BERT Training Loss
![BERTTrainingloss](https://raw.githubusercontent.com/thamizhannal/END3.0/main/Session11-BERT%20and%20BART/Task1_Train_BERT_On_Squad_Dataset/imgs/BERT_Training_loss.png)
### BERT Validation
![BERTValidation](https://raw.githubusercontent.com/thamizhannal/END3.0/main/Session11-BERT%20and%20BART/Task1_Train_BERT_On_Squad_Dataset/imgs/BERT_eval.png)
### BERT Sample Text Prediction
![BERT_SampleTextPrediction](https://raw.githubusercontent.com/thamizhannal/END3.0/main/Session11-BERT%20and%20BART/Task1_Train_BERT_On_Squad_Dataset/imgs/BERT_output.png)
