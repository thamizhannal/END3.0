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
This file is tab seperated file.
Column1 : ArticleTitle
**Column2 : Question
Column3 : Answer**
Column4 : DifficultyFromQuestioner
Column5 : DifficultyFromAnswerer
Column6 : ArticleFile

Question and Answers are the columns that needs to be extracted and pre-processed for this problem. 

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
