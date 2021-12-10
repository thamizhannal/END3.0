### Assignment
 
Assignment (300 points): <br>
Train model we wrote in the class on the following two datasets taken from this link (Links to an external site.): <br>
http://www.cs.cmu.edu/~ark/QA-data/ (Links to an external site.) <br>
https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs (Links to an external site.) <br>

Once done, please upload the file to GitHub and proceed to share the things asked below: <br>
Share the link to your GitHub repo (100 pts for code quality/file structure/model accuracy) (100 pts) <br>
Share the link to your readme file (100 points for proper readme file) <br>
Copy-paste the code related to your dataset preparation for both datasets.  (100 pts) <br>
If your model trains and gets to respectable accuracy (200 pts). <br>
Please remember that the objective of this assignment is to learn how to write code step by steps, so I should be seeing your exploration steps.<br>

## 1. Question-Answer Dataset
This page provides a link to a corpus of Wikipedia articles, manually-generated factoid questions from them, and manually-generated answers to these questions, for use in academic research. These data were collected by Noah Smith, Michael Heilman, Rebecca Hwa, Shay Cohen, Kevin Gimpel, and many students at Carnegie Mellon University and the University of Pittsburgh between 2008 and 2010.

#### Data Format:
This file is tab seperated file. <br>

Column1 : ArticleTitle  <br>
**Column2 : Question <br>
Column3 : Answer** <br>
Column4 : DifficultyFromQuestioner <br>
Column5 : DifficultyFromAnswerer <br>
Column6 : ArticleFile <br>

Question and Answers are the columns that needs to be extracted and pre-processed for this problem.  <br>

## Data Preparation: Question-Answer

#### Download and decompress Question Answer dataset
```python
!wget http://www.cs.cmu.edu/~ark/QA-data/data/Question_Answer_Dataset_v1.2.tar.gz
!tar -zxvf Question_Answer_Dataset_v1.2.tar.gz
```

#### Reading sample data
```python
input_file1 = '/content/Question_Answer_Dataset_v1.2/S10/question_answer_pairs.txt'
lines = open(input_file1, encoding='ISO-8859-1').read().strip().split('\n')
for line in lines[1:21]:
  #print(line)
  print(line.split('\t')[1:3])
```
#### Input File Data format:

ArticleTitle    Question    Answer    DifficultyFromQuestioner    DifficultyFromAnswerer    ArticleFile'
ArticleTitle - index=0 Question - index=1 Answer - index=2
Since this was Question & Answering model, we are intersted in index=1 and index=2 only

```python
class Lang:
  def __init__(self, name):
    self.name = name
    self.word2index = {}
    self.word2count = {}
    self.index2word = {0: "SOS", 1 : "EOS"}
    self.n_words = 2

  def addSentence(self, sentence):
    for word in sentence.split(' '):
      self.addWord(word)

  def addWord(self, word):
    if word not in self.word2index:
      self.word2index[word] = self.n_words
      self.index2word[self.n_words] = word
      self.word2count[word] = 1
      self.n_words += 1
    else:
      self.word2count[word] += 1

# Turn a Unicode string to Plain ASCII. 
def unicodeToAscii(s):
  return ''.join(
      c for c in unicodedata.normalize('NFD', s)
      if unicodedata.category(c) != 'Mn'
  )

#lowercase, trim, and remove non-letter characters
def normalizeString(s):
  s = unicodeToAscii(s.lower().strip())
  s = re.sub(r"([.!?])", r" \1",s)
  s = re.sub(r"[^a-zA-Z.!?]+", r" ",s)       
  return s

def read_input_data(src, dest, root_dir, sub_dir, file_name, reverse=False):
  input_data_pairs = []

  for dir in sub_dir:
    path= '%s/%s/%s' %(root_dir,dir,file_name)
    print(path)

    # utf-8 format was causing reading error, so using ISO-8859-1 format.
    lines = open(path, encoding='ISO-8859-1').read().strip().split('\n')
    
    # Split every line into pairs and normalize
    # Column1: Question
    # Column2: Answer
    # Skip header from reading or 1st line.
    pairs = [[normalizeString(s) for s in line.split('\t')[1:3] ] for line in lines[1:]]
    input_data_pairs.extend(pairs)

  input_data = Lang(src)
  output_data = Lang(dest)

  return input_data, output_data, input_data_pairs

MAX_LENGTH = 10


def filterPair(p):
  print(p)
  return len(p[0].split(' ')) < MAX_LENGTH and \
    len(p[1].split(' ')) < MAX_LENGTH

def filterPairs(pairs):
  return [pair for pair in pairs if filterPair(pair)]

def prepare_data(src, dest, root_dir, sub_dir, file_name, reverse=False):
  
  input_data, output_data, pairs = read_input_data(src, dest, root_dir, sub_dir, file_name, True)

  print("Read %s sentence pairs" %len(pairs))
  pairs = filterPairs(pairs)
  
  print("Trimmed to %s sententce pairs" % len(pairs))
  
  print("Counting words...")
  for pair in pairs:
    input_data.addSentence(pair[0])
    output_data.addSentence(pair[1])
  print("counted words:")
  print(input_data.name, input_data.n_words)
  print(output_data.name, output_data.n_words)
  return input_data, output_data, pairs
```
#### Loading Question answer data from sub directory S08, S09, S10.
```python
input_file1 = '/content/Question_Answer_Dataset_v1.2/S08/question_answer_pairs.txt'
root_dir = '/content/Question_Answer_Dataset_v1.2'
sub_dir = ['S08', 'S09', 'S10']
file_name = 'question_answer_pairs.txt'
src = 'Question'
dest = 'Answer'

EOS_Token = 1
SOS_Token = 0


input_data, output_data, pairs = prepare_data(src, dest, root_dir, sub_dir, file_name, False)
print(random.choice(pairs))
```
#### Print index to every word in dictionary
```python
input_data.index2word
{0: 'SOS',
 1: 'EOS',
 2: 'did',
 3: 'lincoln',
 ....
  997: 'napoleon',
 998: 'rule',
 999: 'profession',
 ...}
```


### Print word to index
```python
input_data.word2index
{'did': 2,
 'lincoln': 3,
 'sign': 4,
 'the': 5,
 ...
 'profession': 999,
 'ruled': 1000,
 'austria': 1001,
 ...}
```

#### Number of words
```python
input_data.n_words
```

### Print a random sample
```python
sample = random.choice(pairs)
sample
```

```python
input_sentence = sample[0]
output_sentence = sample[1]
# print a index of a word from sample.
input_data.word2index['fungi']
```

```python
for word in input_sentence.split(' '):
  print(word)
who
was
blaise
pascal
s
father
?
for word in output_sentence.split(' '):
  print(word)
a
tienne
pascal
.
input_indices = [input_data.word2index[word] for word in input_sentence.split(' ')]
output_indices = [output_data.word2index[word] for word in output_sentence.split(' ')]
input_indices, output_indices
([26, 18, 1065, 1066, 19, 560, 10], [101, 986, 987, 3])
input_indices.append(EOS_Token)
output_indices.append(EOS_Token)
input_indices, output_indices
([26, 18, 1065, 1066, 19, 560, 10, 1], [101, 986, 987, 3, 1])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device
device(type='cuda')
embedding = nn.Embedding(input_size, hidden_size).to(device)
gru = nn.GRU(hidden_size, hidden_size ).to(device)
```

## The Seq2Seq Model

The seq2seq network, or [Encoder Decoder network](https://arxiv.org/pdf/1406.1078v3.pdf), is a model consist of two RNNs as called encoder and decoders. This model accepts word/characters as input sequences (E.g Questions) and output sequence of word/characters (E,g Answers). 

Here Encoder reads the input sequences and construct a hidden vector that holds context present in the input sequences in the condensed format and Decoder understands the context and generates the output sequence according to that.

### The Encoder

The encoder of a seq2seq network is a RNN that outputs some value for every word from the input sentence. For every input word the encoder outputs a vector and a hidden state, and uses the hidden state for the next input word.

![img](https://pytorch.org/tutorials/_images/encoder-network.png)

### The Decoder

The decoder is another RNN that takes the encoder output vector(s) and outputs a sequence of words to create the translation.

In the simplest seq2seq decoder we use only last output of the encoder. This last output is sometimes called the *context vector* as it encodes context from the entire sequence. This context vector is used as the initial hidden state of the decoder.



![img](https://pytorch.org/tutorials/_images/decoder-network.png)





#### Attention Decoder

Attention allows the decoder network to “focus” on a different part of the encoder’s outputs for every step of the decoder’s own outputs. First we calculate a set of *attention weights*. These will be multiplied by the encoder output vectors to create a weighted combination. The result (called `attn_applied` in the code) should contain information about that specific part of the input sequence, and thus help the decoder choose the right output words.

![image](https://miro.medium.com/max/1838/1*tXchCn0hBSUau3WO0ViD7w.jpeg)


## 2. Quora Duplicate Questions 

### DataSet Description:
A set of Quora questions to determine whether pairs of question texts actually correspond to semantically equivalent queries. More than 400,000 lines of potential questions duplicate question pairs.

Objective is problem of identifying duplicate questions.

#### Download Quora Question
```python
!wget http://qim.fs.quoracdn.net/quora_duplicate_questions.tsv
csv_file = 'quora_duplicate_questions.tsv'
```
### Read csv file as pandas
```python
# csv file can be easily manipulated using pandas
import pandas as pd
csv_df = pd.read_csv('quora_duplicate_questions.tsv', sep="\t")
csv_df

	id	qid1	qid2	question1	question2	is_duplicate
0	0	1	2	What is the step by step guide to invest in sh...	What is the step by step guide to invest in sh...	0
1	1	3	4	What is the story of Kohinoor (Koh-i-Noor) Dia...	What would happen if the Indian government sto...	0
2	2	5	6	How can I increase the speed of my internet co...	How can Internet speed be increased by hacking...	0
3	3	7	8	Why am I mentally very lonely? How can I solve...	Find the remainder when [math]23^{24}[/math] i...	0
4	4	9	10	Which one dissolve in water quikly sugar, salt...	Which fish would survive in salt water?	0
```

#### Extract duplicate questions
```python
# Extract duplicate rows or similar rows 
quora_dup_question_df = csv_df[csv_df['is_duplicate'] == 1]
quora_dup_question_df
quora_dup_question_df["question1"]

5         Astrology: I am a Capricorn Sun Cap moon and c...
7                            How can I be a good geologist?
11              How do I read and find my YouTube comments?
12                     What can make Physics easy to learn?
13              What was your first sexual experience like?
```
### Identify vocabulary size 

```python
quora_dup_question_df["question2"].str.len().max

<bound method Series.max of 5         90
7         41
11        38
12        39
13        38
          ..
404280    55
404281    68
404282    47
404284    51
404286    42
Name: question2, Length: 149263, dtype: int64>
```

#### Extract duplicate question and save it in a seperate file
```python
# Save quora duplicate question as csv file.
quora_dup_question_df.to_csv('quora_duplicate_questions.csv')
```
#### Load duplicate file and display sample records
```python
input_file1 = '/content/quora_duplicate_questions.csv'
lines = open(input_file1, encoding='ISO-8859-1').read().strip().split('\n')
for line in lines[0:2]:
  print(line.split(',')[4:6])
  
['question1', 'question2']
['Astrology: I am a Capricorn Sun Cap moon and cap rising...what does that say about me?', '"I\'m a triple Capricorn (Sun']
```
#### Data preprocessing and normalization
```python
class Lang:
  def __init__(self, name):
    self.name = name
    self.word2index = {}
    self.word2count = {}
    self.index2word = {0: "SOS", 1 : "EOS"}
    self.n_words = 2

  def addSentence(self, sentence):
    for word in sentence.split(' '):
      self.addWord(word)

  def addWord(self, word):
    if word not in self.word2index:
      self.word2index[word] = self.n_words
      self.index2word[self.n_words] = word
      self.word2count[word] = 1
      self.n_words += 1
    else:
      self.word2count[word] += 1

# Turn a Unicode string to Plain ASCII. 
def unicodeToAscii(s):
  return ''.join(
      c for c in unicodedata.normalize('NFD', s)
      if unicodedata.category(c) != 'Mn'
  )

#lowercase, trim, and remove non-letter characters
def normalizeString(s):
  s = unicodeToAscii(s.lower().strip())
  s = re.sub(r"([.!?])", r" \1",s)
  s = re.sub(r"[^a-zA-Z.!?]+", r" ",s)       
  return s

def read_input_data(src, dest, input_file_path, reverse=False):
  input_data_pairs = []

  # utf-8 format was causing reading error, so using ISO-8859-1 format.
  lines = open(input_file_path, encoding='ISO-8859-1').read().strip().split('\n')
    
  # Split every line into pairs and normalize
  # Column1: Question
  # Column2: Answer
  # Skip header from reading or 1st line.
  pairs = [[normalizeString(s) for s in line.split(sep=',')[4:6] ] for line in lines[1:]]
  input_data_pairs.extend(pairs)

  input_data = Lang(src)
  output_data = Lang(dest)

  return input_data, output_data, input_data_pairs

def readLangs(lang1, lang2, input_file_path, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open(input_file_path, encoding='utf-8').read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines[1:]]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


MAX_LENGTH = 10

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

def filterPair(p):
  print(p)
  return len(p[0].split(' ')) < MAX_LENGTH and \
    len(p[1].split(' ')) < MAX_LENGTH

def filterPairs(pairs):
  return [pair for pair in pairs if filterPair(pair)]

def prepare_data(question1, question2, input_file_path, reverse=False):
  input_data, output_data, pairs = read_input_data(question1, question2, input_file_path, reverse)

  print("Read %s sentence pairs" %len(pairs))
  pairs = filterPairs(pairs)
  
  print("Trimmed to %s sententce pairs" % len(pairs))
  
  print("Counting words...")
  for pair in pairs:
    input_data.addSentence(pair[0])
    output_data.addSentence(pair[1])
  print("counted words:")
  print(input_data.name, input_data.n_words)
  print(output_data.name, output_data.n_words)
  return input_data, output_data, pairs
```
#### loading questions

```python
input_file_path = '/content/quora_duplicate_questions.csv'
question1 = 'Question1'
question2 = 'Question2'

EOS_Token = 1
SOS_Token = 0


input_data, output_data, pairs = prepare_data(question1, question2, input_file_path, False)
print(random.choice(pairs))
```
#### Display sample questions
```python
pairs[0:5]

[['what can make physics easy to learn ?',
  'how can you make physics easy to learn ?'],
 ['what was your first sexual experience like ?',
  'what was your first sexual experience ?'],
 ['what does manipulation mean ?', 'what does manipulation means ?'],
 ['why do rockets look white ?',
  'why are rockets and boosters painted white ?'],
 ['how do we prepare for upsc ?', 'how do i prepare for civil service ?']]
[ ]

```
#### Get Input size & Hidden size
Hidden size depends on GPU and memory availability

```python
input_size = input_data.n_words
hidden_size = 256
input_size
```


## Final Result Summary:
### Quora Question Similarities: 
Question Similarity Encoder and decoder model with Attention network was build and trined. Random output of 10 samples showed that model accuracy was 90%.

### Wiki Question Answer Dataset
Wiki Question answer data set was trained using Encoder Decoder attention network. Random output of 10 samples results shows model accuracy was 80%.
