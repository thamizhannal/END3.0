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

![images](https://raw.githubusercontent.com/thamizhannal/END3.0/main/Session%2013%20-%20AWS%20Sagemaker%20and%20large-scale%20Model%20Training/imgs/data_distribution.png)

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

![images](https://raw.githubusercontent.com/thamizhannal/END3.0/main/Session%2013%20-%20AWS%20Sagemaker%20and%20large-scale%20Model%20Training/imgs/mode_train_logs.png)
```python
2022-02-16 17:14:13 Starting - Starting the training job.......
2022-02-16 17:14:50 Starting - Preparing the instances for training..........
2022-02-16 17:15:47 Downloading - Downloading input data...
2022-02-16 17:16:08 Training - Downloading the training image.......................................
2022-02-16 17:19:30 Training - Training image download completed. Training in progress..........................................................................................................................................................................................................................................................................................................................................................................................
2022-02-16 17:51:10 Uploading - Uploading generated training model....................................................................................................................................................................................
2022-02-16 18:06:18 Completed - Training job completed
{'TrainingJobName': 'finetune-distilbert-base-cased-2022-02-16-17-13-50',
 'TrainingJobArn': 'arn:aws:sagemaker:us-east-2:454124392436:training-job/finetune-distilbert-base-cased-2022-02-16-17-13-50',
 'ModelArtifacts': {'S3ModelArtifacts': 's3://sagemaker-us-east-2-454124392436/finetune-distilbert-base-cased-2022-02-16-17-13-50/output/model.tar.gz'},
 'TrainingJobStatus': 'Completed',
 'SecondaryStatus': 'Completed',
 'HyperParameters': {'epochs': '5',
  'model_name': '"distilbert-base-cased"',
  'output_dir': '"/opt/ml/checkpoints"',
  'sagemaker_container_log_level': '20',
  'sagemaker_job_name': '"finetune-distilbert-base-cased-2022-02-16-17-13-50"',
  'sagemaker_program': '"train.py"',
  'sagemaker_region': '"us-east-2"',
  'sagemaker_submit_directory': '"s3://sagemaker-us-east-2-454124392436/finetune-distilbert-base-cased-2022-02-16-17-13-50/source/sourcedir.tar.gz"',
  'tokenizer_name': '"distilbert-base-cased"',
  'train_batch_size': '32'},
 'AlgorithmSpecification': {'TrainingImage': '763104351884.dkr.ecr.us-east-2.amazonaws.com/huggingface-pytorch-training:1.7-transformers4.6-gpu-py36-cu110-ubuntu18.04',
  'TrainingInputMode': 'File',
  'MetricDefinitions': [{'Name': 'loss',
    'Regex': "'loss': ([0-9]+(.|e\\-)[0-9]+),?"},
   {'Name': 'learning_rate',
    'Regex': "'learning_rate': ([0-9]+(.|e\\-)[0-9]+),?"},
   {'Name': 'eval_loss', 'Regex': "'eval_loss': ([0-9]+(.|e\\-)[0-9]+),?"},
   {'Name': 'eval_accuracy',
    'Regex': "'eval_accuracy': ([0-9]+(.|e\\-)[0-9]+),?"},
   {'Name': 'eval_f1', 'Regex': "'eval_f1': ([0-9]+(.|e\\-)[0-9]+),?"},
   {'Name': 'eval_precision',
    'Regex': "'eval_precision': ([0-9]+(.|e\\-)[0-9]+),?"},
   {'Name': 'eval_recall', 'Regex': "'eval_recall': ([0-9]+(.|e\\-)[0-9]+),?"},
   {'Name': 'eval_runtime',
    'Regex': "'eval_runtime': ([0-9]+(.|e\\-)[0-9]+),?"},
   {'Name': 'eval_samples_per_second',
    'Regex': "'eval_samples_per_second': ([0-9]+(.|e\\-)[0-9]+),?"},
   {'Name': 'epoch', 'Regex': "'epoch': ([0-9]+(.|e\\-)[0-9]+),?"}],
  'EnableSageMakerMetricsTimeSeries': True},
 'RoleArn': 'arn:aws:iam::454124392436:role/service-role/AmazonSageMaker-ExecutionRole-20220201T224378',
 'InputDataConfig': [{'ChannelName': 'train',
   'DataSource': {'S3DataSource': {'S3DataType': 'S3Prefix',
     'S3Uri': 's3://sagemaker-us-east-2-454124392436/sagemaker/datasets/tweets_hate_speech_detection/train',
     'S3DataDistributionType': 'FullyReplicated'}},
   'CompressionType': 'None',
   'RecordWrapperType': 'None'},
  {'ChannelName': 'test',
   'DataSource': {'S3DataSource': {'S3DataType': 'S3Prefix',
     'S3Uri': 's3://sagemaker-us-east-2-454124392436/sagemaker/datasets/tweets_hate_speech_detection/test',
     'S3DataDistributionType': 'FullyReplicated'}},
   'CompressionType': 'None',
   'RecordWrapperType': 'None'}],
 'OutputDataConfig': {'KmsKeyId': '',
  'S3OutputPath': 's3://sagemaker-us-east-2-454124392436/'},
 'ResourceConfig': {'InstanceType': 'ml.p3.2xlarge',
  'InstanceCount': 2,
  'VolumeSizeInGB': 30},
 'StoppingCondition': {'MaxRuntimeInSeconds': 36000},
 'CreationTime': datetime.datetime(2022, 2, 16, 17, 14, 13, 461000, tzinfo=tzlocal()),
 'TrainingStartTime': datetime.datetime(2022, 2, 16, 17, 15, 47, 782000, tzinfo=tzlocal()),
 'TrainingEndTime': datetime.datetime(2022, 2, 16, 18, 6, 18, 5000, tzinfo=tzlocal()),
 'LastModifiedTime': datetime.datetime(2022, 2, 16, 18, 6, 18, 5000, tzinfo=tzlocal()),
 'SecondaryStatusTransitions': [{'Status': 'Starting',
   'StartTime': datetime.datetime(2022, 2, 16, 17, 14, 13, 461000, tzinfo=tzlocal()),
   'EndTime': datetime.datetime(2022, 2, 16, 17, 15, 47, 782000, tzinfo=tzlocal()),
   'StatusMessage': 'Preparing the instances for training'},
  {'Status': 'Downloading',
   'StartTime': datetime.datetime(2022, 2, 16, 17, 15, 47, 782000, tzinfo=tzlocal()),
   'EndTime': datetime.datetime(2022, 2, 16, 17, 16, 8, 568000, tzinfo=tzlocal()),
   'StatusMessage': 'Downloading input data'},
  {'Status': 'Training',
   'StartTime': datetime.datetime(2022, 2, 16, 17, 16, 8, 568000, tzinfo=tzlocal()),
   'EndTime': datetime.datetime(2022, 2, 16, 17, 51, 10, 623000, tzinfo=tzlocal()),
   'StatusMessage': 'Training image download completed. Training in progress.'},
  {'Status': 'Uploading',
   'StartTime': datetime.datetime(2022, 2, 16, 17, 51, 10, 623000, tzinfo=tzlocal()),
   'EndTime': datetime.datetime(2022, 2, 16, 18, 6, 18, 5000, tzinfo=tzlocal()),
   'StatusMessage': 'Uploading generated training model'},
  {'Status': 'Completed',
   'StartTime': datetime.datetime(2022, 2, 16, 18, 6, 18, 5000, tzinfo=tzlocal()),
   'EndTime': datetime.datetime(2022, 2, 16, 18, 6, 18, 5000, tzinfo=tzlocal()),
   'StatusMessage': 'Training job completed'}],
 'FinalMetricDataList': [{'MetricName': 'loss',
   'Value': 0.003599999938160181,
   'Timestamp': datetime.datetime(2022, 2, 16, 17, 49, 13, tzinfo=tzlocal())},
  {'MetricName': 'learning_rate',
   'Value': 2.3809523582458496,
   'Timestamp': datetime.datetime(2022, 2, 16, 17, 49, 13, tzinfo=tzlocal())},
  {'MetricName': 'eval_loss',
   'Value': 0.1793152242898941,
   'Timestamp': datetime.datetime(2022, 2, 16, 17, 50, 44, tzinfo=tzlocal())},
  {'MetricName': 'eval_accuracy',
   'Value': 0.9712499976158142,
   'Timestamp': datetime.datetime(2022, 2, 16, 17, 50, 44, tzinfo=tzlocal())},
  {'MetricName': 'eval_f1',
   'Value': 0.7704590559005737,
   'Timestamp': datetime.datetime(2022, 2, 16, 17, 50, 44, tzinfo=tzlocal())},
  {'MetricName': 'eval_precision',
   'Value': 0.8109243512153625,
   'Timestamp': datetime.datetime(2022, 2, 16, 17, 50, 44, tzinfo=tzlocal())},
  {'MetricName': 'eval_recall',
   'Value': 0.73384028673172,
   'Timestamp': datetime.datetime(2022, 2, 16, 17, 50, 44, tzinfo=tzlocal())},
  {'MetricName': 'eval_runtime',
   'Value': 20.123600006103516,
   'Timestamp': datetime.datetime(2022, 2, 16, 17, 50, 44, tzinfo=tzlocal())},
  {'MetricName': 'eval_samples_per_second',
   'Value': 198.77099609375,
   'Timestamp': datetime.datetime(2022, 2, 16, 17, 50, 44, tzinfo=tzlocal())},
  {'MetricName': 'epoch',
   'Value': 5.0,
   'Timestamp': datetime.datetime(2022, 2, 16, 17, 50, 44, tzinfo=tzlocal())}],
 'EnableNetworkIsolation': False,
 'EnableInterContainerTrafficEncryption': False,
 'EnableManagedSpotTraining': False,
 'TrainingTimeInSeconds': 3031,
 'BillableTimeInSeconds': 3031,
 'DebugHookConfig': {'S3OutputPath': 's3://sagemaker-us-east-2-454124392436/',
  'CollectionConfigurations': []},
 'ProfilerConfig': {'S3OutputPath': 's3://sagemaker-us-east-2-454124392436/',
  'ProfilingIntervalInMilliseconds': 500},
 'ProfilerRuleConfigurations': [{'RuleConfigurationName': 'ProfilerReport-1645031653',
   'RuleEvaluatorImage': '915447279597.dkr.ecr.us-east-2.amazonaws.com/sagemaker-debugger-rules:latest',
   'VolumeSizeInGB': 0,
   'RuleParameters': {'rule_to_invoke': 'ProfilerReport'}}],
 'ProfilerRuleEvaluationStatuses': [{'RuleConfigurationName': 'ProfilerReport-1645031653',
   'RuleEvaluationJobArn': 'arn:aws:sagemaker:us-east-2:454124392436:processing-job/finetune-distilbert-base-c-profilerreport-1645031653-799b88e2',
   'RuleEvaluationStatus': 'InProgress',
   'LastModifiedTime': datetime.datetime(2022, 2, 16, 18, 6, 6, 456000, tzinfo=tzlocal())}],
 'ProfilingStatus': 'Enabled',
 'ResponseMetadata': {'RequestId': 'e47fbb07-4c82-4618-81d0-3201a70e9e22',
  'HTTPStatusCode': 200,
  'HTTPHeaders': {'x-amzn-requestid': 'e47fbb07-4c82-4618-81d0-3201a70e9e22',
   'content-type': 'application/x-amz-json-1.1',
   'content-length': '5696',
   'date': 'Wed, 16 Feb 2022 18:06:21 GMT'},
  'RetryAttempts': 0}}
  ```
## Evaluation metrics:
![images](https://raw.githubusercontent.com/thamizhannal/END3.0/main/Session%2013%20-%20AWS%20Sagemaker%20and%20large-scale%20Model%20Training/imgs/metrics.png)

## Evaluation Ouput:

![images](https://raw.githubusercontent.com/thamizhannal/END3.0/main/Session%2013%20-%20AWS%20Sagemaker%20and%20large-scale%20Model%20Training/imgs/model_eval_results.png)

## Endpoints
![images](https://raw.githubusercontent.com/thamizhannal/END3.0/main/Session%2013%20-%20AWS%20Sagemaker%20and%20large-scale%20Model%20Training/imgs/endpoint_link.png)

## References
1. Hugging Face Twitter hate speech dataset https://huggingface.co/datasets/tweets_hate_speech_detection
2. AWS Sage Maker https://aws.amazon.com/about-aws/whats-new/2017/11/introducing-amazon-sagemaker/
3. https://huggingface.co/docs/datasets/_modules/datasets/dataset_dict.html
