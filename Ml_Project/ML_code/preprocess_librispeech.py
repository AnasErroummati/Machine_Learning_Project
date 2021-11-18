#Dividing the available data into separate datasets (e.g., training and validation datasets) and then convert the datasets into TFRecord files containing the data represented as tf.Example data structures:  



from absl import app, logging, flags  

import os 

import json 

import tensorflow as tf 

 
 

from utils import preprocessing, encoding 

from utils.data import librispeech 

from hparams import * 

 
 
 

FLAGS = flags.FLAGS 

 
 # The flags tell you what arguments you pass to preprocess_librispeech.py  in our case we used:
#python3 preprocess_librispeech.py --data_dir ../../../mnt/d/LibriSpeech --output_dir ...

flags.DEFINE_string( 

    'data_dir', None, 

    'Directory to read Librispeech data from.') 

flags.DEFINE_string( 

    'output_dir', './data', 

    'Directory to save preprocessed data.') 

#Optional Flags

flags.DEFINE_integer( 

    'max_length', 0, 

    'Max audio length in seconds.') 

 
 
 # This function is used to write the converted data into tf record format using the TFRecord writer.

def write_dataset(dataset, name): 

 
 

    filepath = os.path.join(FLAGS.output_dir, 

        '{}.tfrecord'.format(name)) 

 
 

    writer = tf.data.experimental.TFRecordWriter(filepath) 

    writer.write(dataset) 

 
 

    logging.info('Wrote {} dataset to {}'.format( 

        name, filepath)) 

 
 
 

def main(_): 

 

    hparams = { 

 
 

        HP_TOKEN_TYPE: HP_TOKEN_TYPE.domain.values[1], 

        HP_VOCAB_SIZE: HP_VOCAB_SIZE.domain.values[0], 

 
 

        # Preprocessing 

        HP_MEL_BINS: HP_MEL_BINS.domain.values[0], 

        HP_FRAME_LENGTH: HP_FRAME_LENGTH.domain.values[0], 

        HP_FRAME_STEP: HP_FRAME_STEP.domain.values[0], 

        HP_HERTZ_LOW: HP_HERTZ_LOW.domain.values[0], 

        HP_HERTZ_HIGH: HP_HERTZ_HIGH.domain.values[0], 

        HP_DOWNSAMPLE_FACTOR: HP_DOWNSAMPLE_FACTOR.domain.values[0] 

 
 

    } 

 
 # These commands mean that the splits will be applied to the folder that contains the lables and audios wich is the dev-clean floder.

    train_splits = [ 

        'dev-clean' 

    ] 

 
 

    dev_splits = [ 

        'dev-clean' 

    ] 

 
 

    test_splits = [ 

        'dev-clean' 

    ] 

 
 
 
  #Load hyper parameters that were specified above to the variable _hparams

    _hparams = {k.name: v for k, v in hparams.items()} 

 

    texts_gen = librispeech.texts_generator(FLAGS.data_dir, 

        split_names=train_splits) 

 
 
# function necessary in order to generate the encoder.subwords file that contains all the labels, this file will be used for training, validation and testing

    encoder_fn, decoder_fn, vocab_size = encoding.get_encoder( 

        encoder_dir=FLAGS.output_dir, 

        hparams=_hparams, 

        texts_generator=texts_gen) 

    _hparams[HP_VOCAB_SIZE.name] = vocab_size 

 
 
# Loading datasets

    train_dataset = librispeech.load_dataset( 

        FLAGS.data_dir, train_splits) 

    dev_dataset = librispeech.load_dataset( 

        FLAGS.data_dir, dev_splits) 

    test_dataset = librispeech.load_dataset( 

        FLAGS.data_dir, test_splits) 

 
 
# preprocessing each dataset the size of the dataset is following the 80/10/10 distribution.

    train_dataset = preprocessing.preprocess_dataset( 

        train_dataset, 

        encoder_fn=encoder_fn, 

        hparams=_hparams, 

        max_length=FLAGS.max_length, 

        save_plots=True) 

    write_dataset(train_dataset, 'train') 

 
 

    dev_dataset = preprocessing.preprocess_dataset( 

        dev_dataset, 

        encoder_fn=encoder_fn, 

        hparams=_hparams, 

        max_length=FLAGS.max_length) 

    write_dataset(dev_dataset, 'dev') 

 
 

    test_dataset = preprocessing.preprocess_dataset( 

        test_dataset, 

        encoder_fn=encoder_fn, 

        hparams=_hparams, 

        max_length=FLAGS.max_length) 

    write_dataset(test_dataset, 'test') 

 
 
 

if __name__ == '__main__': 

 
 

    flags.mark_flag_as_required('data_dir') 

 
 

    app.run(main) 

 