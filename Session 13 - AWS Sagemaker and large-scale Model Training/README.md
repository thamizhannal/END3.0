# AWS Sagemaker and large-scale Model Training

# Assignment
 
1. Go through this notebook (Links to an external site.), change the dataset (to anything other than "Amazon Reviews Polarity Dataset"). <br>
https://github.com/aws-samples/finetune-deploy-bert-with-amazon-sagemaker-for-hugging-face/blob/main/finetune-distilbert.ipynb <br>
2. Record the whole step using any video capturing software (like OBS). <br>
3. Upload the video on youtube (you can make it unlisted, but allow for embedding) <br>
4. Share the link to your YouTube video, and the GitHub link where I can see the code you used for training (move the notebook with logs from Amazon to Github) <br>

## YouTube Recorded Video Link:

## Introduction to AWS SageMaker
Amazon SageMaker is a fully-managed service that enables data scientists and developers to quickly and easily build, train, and deploy machine learning models at any scale. Amazon SageMaker includes modules that can be used together or independently to build, train, and deploy your machine learning models.

**Build**
Amazon SageMaker makes it easy to build ML models and get them ready for training by providing everything you need to quickly connect to your training data, and to select and optimize the best algorithm and framework for your application.

**Train**
You can begin training your model with a single click in the Amazon SageMaker console. Amazon SageMaker manages all of the underlying infrastructure for you and can easily scale to train models at petabyte scale.

**Deploy**
Once your model is trained and tuned, Amazon SageMaker makes it easy to deploy in production so you can start running generating predictions on new data (a process called inference)


## Introduction to Hugging Face:
The AI community building the future.
Build, train and deploy state of the art models powered by the reference open source in machine learning.

## What is Hugging Face Transformer?
Image result for hugging face
The Hugging Face transformers package is an immensely popular Python library providing pretrained models that are extraordinarily useful for a variety of natural language processing (NLP) tasks. It previously supported only PyTorch, but, as of late 2019, TensorFlow 2 is supported as well.


## DataSet Description:

**DataSet** <br>
Total num of records: 31962 <br>
Train: 80% of total records: 25000 <br>
Evaluation: 20% of total records: 6961 <br>

![image](https://user-images.githubusercontent.com/8234814/154300892-0062433d-73ef-463f-9d42-7f1c0cf10ba4.png)

## AWS Training:
```python
tokenizer_name = 'distilbert-base-cased'

# Set the format to PyTorch
train_dataset = train_dataset.rename_column("label", "labels")
train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
test_dataset = test_dataset.rename_column("label", "labels")
test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

# Hyper parameters
hyperparameters={'epochs': 10,
                 'train_batch_size': 32,
                 'model_name': model_name,
                 'tokenizer_name': tokenizer_name,
                 'output_dir':'/opt/ml/checkpoints',
                 }
# huggingface_estimator
huggingface_estimator = HuggingFace(entry_point='train.py',
                            source_dir='./scripts',
                            instance_type='ml.p3.2xlarge',
                            instance_count=2,
                            role=role,
                            transformers_version='4.6', 
                            pytorch_version='1.7',
                            py_version='py36',
                            hyperparameters = hyperparameters,
                            #use_spot_instances=use_spot_instances,
                            #max_wait=36000,
                            metric_definitions=metric_definitions,
                            max_run=36000, # expected max run in seconds
                        )
                       
huggingface_estimator.fit({'train': training_input_path, 'test': test_input_path}, wait=False, job_name=training_job_name )
                       
```

## Training Logs:


## Evaluation:

## parameters:
## Results:

## References
1. Hugging Face Twitter hate speech dataset https://huggingface.co/datasets/tweets_hate_speech_detection
2. AWS Sage Maker https://aws.amazon.com/about-aws/whats-new/2017/11/introducing-amazon-sagemaker/
3. 
