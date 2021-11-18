from argparse import ArgumentParser
import os

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)

from utils import preprocessing, encoding, decoding
from utils import model as model_utils
from model import build_keras_model
from hparams import *


def main(args):

# load the checkpoint
    model_dir = os.path.dirname(os.path.realpath(args.checkpoint))

# load the hyperparameters
    hparams = model_utils.load_hparams(model_dir)

    encode_fn, tok_to_text, vocab_size = encoding.get_encoder(
        encoder_dir=model_dir,
        hparams=hparams)
    hparams[HP_VOCAB_SIZE.name] = vocab_size

#build our model
    model = build_keras_model(hparams)
    model.load_weights(args.checkpoint)

#Preprocess the wav audio file

    audio, sr = preprocessing.tf_load_audio(args.input)

    log_melspec = preprocessing.preprocess_audio(
        audio=audio,
        sample_rate=sr,
        hparams=hparams)
    log_melspec = tf.expand_dims(log_melspec, axis=0)

#decode the audio into text using the model weights and hyperparameter
    decoder_fn = decoding.greedy_decode_fn(model, hparams)

    decoded = decoder_fn(log_melspec)[0]

# returning text
    transcription = tok_to_text(decoded)

    print('Transcription:', transcription.numpy().decode('utf8'))


def parse_args():

    ap = ArgumentParser()

# the expression takes 2 arguments, the checkpoint + the wav file 

    ap.add_argument('--checkpoint', type=str, required=True,
        help='Checkpoint to load.')
    ap.add_argument('-i', '--input', type=str, required=True,
        help='Wav file to transcribe.')

    return ap.parse_args()


if __name__ == '__main__':

    args = parse_args()
    main(args)
