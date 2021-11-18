#Create Satistics from the data + Display schema 

import tensorflow_data_validation as tfdv 
stats = tfdv.generate_statistics_from_tfrecord(data_location='/content/drive/MyDrive/training data/training.tfrecord') 

#Generating Schema from Your Data

schema = tfdv.infer_schema(stats) 
tfdv.display_schema(schema) 

#Comparing Datasets the training.tfrecord contain both the train and test datasets: 

import tensorflow_data_validation as tfdv  

#Generating statistics for both datasets

train_stats = tfdv.generate_statistics_from_tfrecord( 

 data_location='/content/drive/MyDrive/training.tfrecord') 

test_stats = tfdv.generate_statistics_from_tfrecord( 

 data_location='/content/drive/MyDrive/training.tfrecord') 

# Display Statistics

tfdv.visualize_statistics(lhs_statistics=test_stats, rhs_statistics=train_stats, 

 lhs_name='Test_DATASET', rhs_name='TRAIN_DATASET') 

#Anomaly checking based on data staistics: 

anomalies = tfdv.validate_statistics(statistics=train_stats, schema=schema)  

# Dislay Anomalies
tfdv.display_anomalies(anomalies) 