## Session 5 - TorchText & Advanced Concepts

### Assignment

Pick any 2 datasets (except AG_NEWS) from torchtext.datasets and train your model on them achieving 50% more accuracy than random prediction. <br>
Upload to Github with a proper readme file describing your datasets, and showing your logs as well. <br>

### **Common Utility methods:**

**get_tokenizer("basic_english"):** loads basic english tokenizer provided by torchtext.

**build_vocab_from_iterator:** Builds vocabulary from tokenized provided by torchtext.

**collate_batch(batch):** creates label_list, text_list and offset_list from batch of the records. E.g. if batch size is 64, then these list were created appending each records one after another.

**TextClassificationModel:** Simple text classification model using 

**Train, test and validation set:** Train set was taken from 95% of train data and rest of the 5% used for validation purpose. test set was used as it is.

## 1. Sogou News Classification

### Sogou New Dataset Summary
**Description:** The Sogou News dataset is a mixture of 2,909,551 news articles from the SogouCA and SogouCS news corpora, in 5 categories. The number of training samples selected for each class is 90,000 and testing 12,000. <br>
**Data Fields:** content: a string feature. label: a classification label, with possible values including sports (0), finance (1), entertainment (2), automobile (3), technology (4). <br>


**loading Yelp data set:**
train_iter, test_iter = YelpReviewPolarity(split=('train','test'))

**Model Train and validation:**

```python
from torchtext.data.functional import to_map_style_dataset
# Hyperparameters
EPOCHS = 5 # epoch
LR = 5  # learning rate
BATCH_SIZE = 64 # batch size for training
  
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
total_accu = None
train_iter, test_iter = SogouNews()
train_dataset = to_map_style_dataset(train_iter)
test_dataset = to_map_style_dataset(test_iter)
num_train = int(len(train_dataset) * 0.95)
split_train_, split_valid_ = \
    random_split(train_dataset, [num_train, len(train_dataset) - num_train])

train_dataloader = DataLoader(split_train_, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_batch)
valid_dataloader = DataLoader(split_valid_, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=True, collate_fn=collate_batch)

for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
    train(train_dataloader)
    accu_val = evaluate(valid_dataloader)
    if total_accu is not None and total_accu > accu_val:
      scheduler.step()
    else:
       total_accu = accu_val
    print('-' * 59)
    print('| end of epoch {:3d} | time: {:5.2f}s | '
          'valid accuracy {:8.3f} '.format(epoch,
                                           time.time() - epoch_start_time,
                                           accu_val))
    print('-' * 59)
 ```
 ```
 | epoch   1 |   500/ 6680 batches | accuracy    0.825
| epoch   1 |  1000/ 6680 batches | accuracy    0.908
| epoch   1 |  1500/ 6680 batches | accuracy    0.917
| epoch   1 |  2000/ 6680 batches | accuracy    0.921
| epoch   1 |  2500/ 6680 batches | accuracy    0.920
| epoch   1 |  3000/ 6680 batches | accuracy    0.925
| epoch   1 |  3500/ 6680 batches | accuracy    0.923
| epoch   1 |  4000/ 6680 batches | accuracy    0.926
| epoch   1 |  4500/ 6680 batches | accuracy    0.926
| epoch   1 |  5000/ 6680 batches | accuracy    0.929
| epoch   1 |  5500/ 6680 batches | accuracy    0.929
| epoch   1 |  6000/ 6680 batches | accuracy    0.928
| epoch   1 |  6500/ 6680 batches | accuracy    0.930
-----------------------------------------------------------
| end of epoch   1 | time: 146.60s | valid accuracy    0.931 
-----------------------------------------------------------
| epoch   2 |   500/ 6680 batches | accuracy    0.933
| epoch   2 |  1000/ 6680 batches | accuracy    0.932
| epoch   2 |  1500/ 6680 batches | accuracy    0.931
| epoch   2 |  2000/ 6680 batches | accuracy    0.932
| epoch   2 |  2500/ 6680 batches | accuracy    0.930
| epoch   2 |  3000/ 6680 batches | accuracy    0.930
| epoch   2 |  3500/ 6680 batches | accuracy    0.930
| epoch   2 |  4000/ 6680 batches | accuracy    0.933
| epoch   2 |  4500/ 6680 batches | accuracy    0.932
| epoch   2 |  5000/ 6680 batches | accuracy    0.932
| epoch   2 |  5500/ 6680 batches | accuracy    0.934
| epoch   2 |  6000/ 6680 batches | accuracy    0.930
| epoch   2 |  6500/ 6680 batches | accuracy    0.934
-----------------------------------------------------------
| end of epoch   2 | time: 145.69s | valid accuracy    0.932 
-----------------------------------------------------------
| epoch   3 |   500/ 6680 batches | accuracy    0.936
| epoch   3 |  1000/ 6680 batches | accuracy    0.932
| epoch   3 |  1500/ 6680 batches | accuracy    0.932
| epoch   3 |  2000/ 6680 batches | accuracy    0.936
| epoch   3 |  2500/ 6680 batches | accuracy    0.932
| epoch   3 |  3000/ 6680 batches | accuracy    0.933
| epoch   3 |  3500/ 6680 batches | accuracy    0.933
| epoch   3 |  4000/ 6680 batches | accuracy    0.934
| epoch   3 |  4500/ 6680 batches | accuracy    0.934
| epoch   3 |  5000/ 6680 batches | accuracy    0.935
| epoch   3 |  5500/ 6680 batches | accuracy    0.934
| epoch   3 |  6000/ 6680 batches | accuracy    0.935
| epoch   3 |  6500/ 6680 batches | accuracy    0.934
-----------------------------------------------------------
| end of epoch   3 | time: 147.13s | valid accuracy    0.934 
-----------------------------------------------------------
| epoch   4 |   500/ 6680 batches | accuracy    0.936
| epoch   4 |  1000/ 6680 batches | accuracy    0.935
| epoch   4 |  1500/ 6680 batches | accuracy    0.935
| epoch   4 |  2000/ 6680 batches | accuracy    0.936
| epoch   4 |  2500/ 6680 batches | accuracy    0.937
| epoch   4 |  3000/ 6680 batches | accuracy    0.936
| epoch   4 |  3500/ 6680 batches | accuracy    0.934
| epoch   4 |  4000/ 6680 batches | accuracy    0.935
| epoch   4 |  4500/ 6680 batches | accuracy    0.936
| epoch   4 |  5000/ 6680 batches | accuracy    0.934
| epoch   4 |  5500/ 6680 batches | accuracy    0.936
| epoch   4 |  6000/ 6680 batches | accuracy    0.936
| epoch   4 |  6500/ 6680 batches | accuracy    0.935
-----------------------------------------------------------
| end of epoch   4 | time: 148.84s | valid accuracy    0.934 
-----------------------------------------------------------
| epoch   5 |   500/ 6680 batches | accuracy    0.941
| epoch   5 |  1000/ 6680 batches | accuracy    0.939
| epoch   5 |  1500/ 6680 batches | accuracy    0.939
| epoch   5 |  2000/ 6680 batches | accuracy    0.941
| epoch   5 |  2500/ 6680 batches | accuracy    0.941
| epoch   5 |  3000/ 6680 batches | accuracy    0.938
| epoch   5 |  3500/ 6680 batches | accuracy    0.940
| epoch   5 |  4000/ 6680 batches | accuracy    0.940
| epoch   5 |  4500/ 6680 batches | accuracy    0.940
| epoch   5 |  5000/ 6680 batches | accuracy    0.941
| epoch   5 |  5500/ 6680 batches | accuracy    0.941
| epoch   5 |  6000/ 6680 batches | accuracy    0.940
| epoch   5 |  6500/ 6680 batches | accuracy    0.940
-----------------------------------------------------------
| end of epoch   5 | time: 145.18s | valid accuracy    0.936 
-----------------------------------------------------------
```
#### Objective
Pick any 2 datasets (except AG_NEWS) from torchtext.datasets and train your model on them achieving 50% more accuracy than random prediction. Upload to Github with a proper readme file describing your datasets, and showing your logs as well.

#### Result:
SogouNews data set was input data set for classification and this was 5 class classification problem.
Objective was to achieving 50% more accuracy than random prediction(20%) which was 30%.
Highest validation accuracy acheived was 93.6% which was 50% more accuracy than random prediction.


## 2. Yelp Review Polarity Classification:
#### Yelp Data Description:
The Yelp reviews polarity dataset is constructed by considering stars 1 and 2 negative, and 3 and 4 positive. For each polarity 280,000 training samples and 19,000 testing samples are take randomly. In total there are 560,000 trainig samples and 38,000 testing samples. **Negative polarity is class 1, and positive class 2**.

```python
# Load Yelp Reivew Polarity dataset.
train_iter, test_iter = YelpReviewPolarity(split=('train','test'))
```
**Train & test data size:**
Total number of train data:560000
Total number of test data:38000

**Train, test and validation**
```python
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset
# Hyperparameters
EPOCHS = 5 # epoch
LR = 5  # learning rate
BATCH_SIZE = 64 # batch size for training
  
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
total_accu = None
train_iter, test_iter = YelpReviewPolarity()

train_dataset = to_map_style_dataset(train_iter)
test_dataset = to_map_style_dataset(test_iter)
num_train = int(len(train_dataset) * 0.95)
split_train_, split_valid_ = \
    random_split(train_dataset, [num_train, len(train_dataset) - num_train])

train_dataloader = DataLoader(split_train_, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_batch)
valid_dataloader = DataLoader(split_valid_, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=True, collate_fn=collate_batch)

for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
    train(train_dataloader)
    accu_val = evaluate(valid_dataloader)
    if total_accu is not None and total_accu > accu_val:
      scheduler.step()
    else:
       total_accu = accu_val
    print('-' * 59)
    print('| end of epoch {:3d} | time: {:5.2f}s | '
          'valid accuracy {:8.3f} '.format(epoch,
                                           time.time() - epoch_start_time,
                                           accu_val))
    print('-' * 59)
```
```
| epoch   1 |   500/ 8313 batches | accuracy    0.779
| epoch   1 |  1000/ 8313 batches | accuracy    0.863
| epoch   1 |  1500/ 8313 batches | accuracy    0.880
| epoch   1 |  2000/ 8313 batches | accuracy    0.889
| epoch   1 |  2500/ 8313 batches | accuracy    0.896
| epoch   1 |  3000/ 8313 batches | accuracy    0.898
| epoch   1 |  3500/ 8313 batches | accuracy    0.903
| epoch   1 |  4000/ 8313 batches | accuracy    0.900
| epoch   1 |  4500/ 8313 batches | accuracy    0.904
| epoch   1 |  5000/ 8313 batches | accuracy    0.905
| epoch   1 |  5500/ 8313 batches | accuracy    0.907
| epoch   1 |  6000/ 8313 batches | accuracy    0.911
| epoch   1 |  6500/ 8313 batches | accuracy    0.908
| epoch   1 |  7000/ 8313 batches | accuracy    0.907
| epoch   1 |  7500/ 8313 batches | accuracy    0.911
| epoch   1 |  8000/ 8313 batches | accuracy    0.911
-----------------------------------------------------------
| end of epoch   1 | time: 61.64s | valid accuracy    0.912 
-----------------------------------------------------------
| epoch   2 |   500/ 8313 batches | accuracy    0.918
| epoch   2 |  1000/ 8313 batches | accuracy    0.915
| epoch   2 |  1500/ 8313 batches | accuracy    0.917
| epoch   2 |  2000/ 8313 batches | accuracy    0.916
| epoch   2 |  2500/ 8313 batches | accuracy    0.915
| epoch   2 |  3000/ 8313 batches | accuracy    0.917
| epoch   2 |  3500/ 8313 batches | accuracy    0.918
| epoch   2 |  4000/ 8313 batches | accuracy    0.916
| epoch   2 |  4500/ 8313 batches | accuracy    0.918
| epoch   2 |  5000/ 8313 batches | accuracy    0.918
| epoch   2 |  5500/ 8313 batches | accuracy    0.916
| epoch   2 |  6000/ 8313 batches | accuracy    0.920
| epoch   2 |  6500/ 8313 batches | accuracy    0.919
| epoch   2 |  7000/ 8313 batches | accuracy    0.922
| epoch   2 |  7500/ 8313 batches | accuracy    0.921
| epoch   2 |  8000/ 8313 batches | accuracy    0.918
-----------------------------------------------------------
| end of epoch   2 | time: 61.66s | valid accuracy    0.913 
-----------------------------------------------------------
| epoch   3 |   500/ 8313 batches | accuracy    0.925
| epoch   3 |  1000/ 8313 batches | accuracy    0.922
| epoch   3 |  1500/ 8313 batches | accuracy    0.920
| epoch   3 |  2000/ 8313 batches | accuracy    0.921
| epoch   3 |  2500/ 8313 batches | accuracy    0.922
| epoch   3 |  3000/ 8313 batches | accuracy    0.922
| epoch   3 |  3500/ 8313 batches | accuracy    0.921
| epoch   3 |  4000/ 8313 batches | accuracy    0.923
| epoch   3 |  4500/ 8313 batches | accuracy    0.923
| epoch   3 |  5000/ 8313 batches | accuracy    0.923
| epoch   3 |  5500/ 8313 batches | accuracy    0.922
| epoch   3 |  6000/ 8313 batches | accuracy    0.922
| epoch   3 |  6500/ 8313 batches | accuracy    0.923
| epoch   3 |  7000/ 8313 batches | accuracy    0.924
| epoch   3 |  7500/ 8313 batches | accuracy    0.925
| epoch   3 |  8000/ 8313 batches | accuracy    0.924
-----------------------------------------------------------
| end of epoch   3 | time: 61.70s | valid accuracy    0.914 
-----------------------------------------------------------
| epoch   4 |   500/ 8313 batches | accuracy    0.925
| epoch   4 |  1000/ 8313 batches | accuracy    0.925
| epoch   4 |  1500/ 8313 batches | accuracy    0.926
| epoch   4 |  2000/ 8313 batches | accuracy    0.926
| epoch   4 |  2500/ 8313 batches | accuracy    0.928
| epoch   4 |  3000/ 8313 batches | accuracy    0.922
| epoch   4 |  3500/ 8313 batches | accuracy    0.926
| epoch   4 |  4000/ 8313 batches | accuracy    0.926
| epoch   4 |  4500/ 8313 batches | accuracy    0.928
| epoch   4 |  5000/ 8313 batches | accuracy    0.925
| epoch   4 |  5500/ 8313 batches | accuracy    0.925
| epoch   4 |  6000/ 8313 batches | accuracy    0.923
| epoch   4 |  6500/ 8313 batches | accuracy    0.929
| epoch   4 |  7000/ 8313 batches | accuracy    0.927
| epoch   4 |  7500/ 8313 batches | accuracy    0.927
| epoch   4 |  8000/ 8313 batches | accuracy    0.926
-----------------------------------------------------------
| end of epoch   4 | time: 62.15s | valid accuracy    0.927 
-----------------------------------------------------------
| epoch   5 |   500/ 8313 batches | accuracy    0.928
| epoch   5 |  1000/ 8313 batches | accuracy    0.928
| epoch   5 |  1500/ 8313 batches | accuracy    0.929
| epoch   5 |  2000/ 8313 batches | accuracy    0.926
| epoch   5 |  2500/ 8313 batches | accuracy    0.931
| epoch   5 |  3000/ 8313 batches | accuracy    0.929
| epoch   5 |  3500/ 8313 batches | accuracy    0.930
| epoch   5 |  4000/ 8313 batches | accuracy    0.928
| epoch   5 |  4500/ 8313 batches | accuracy    0.929
| epoch   5 |  5000/ 8313 batches | accuracy    0.927
| epoch   5 |  5500/ 8313 batches | accuracy    0.927
| epoch   5 |  6000/ 8313 batches | accuracy    0.929
| epoch   5 |  6500/ 8313 batches | accuracy    0.927
| epoch   5 |  7000/ 8313 batches | accuracy    0.930
| epoch   5 |  7500/ 8313 batches | accuracy    0.926
| epoch   5 |  8000/ 8313 batches | accuracy    0.930
-----------------------------------------------------------
| end of epoch   5 | time: 61.61s | valid accuracy    0.929 
-----------------------------------------------------------
```
Objective:
Pick any 2 datasets (except AG_NEWS) from torchtext.datasets and train your model on them achieving 50% more accuracy than random prediction. Upload to Github with a proper readme file describing your datasets, and showing your logs as well.

Result:
Yelp Review Plarity data set was input data set for classification and this was two class classification problem.
Objective was achieveing 50% more accuracy than random prediction(50%) which was 75%.
Highest validation accuracy acheived was 92.9% which was 50% more accuracy than random prediction.






