from __future__ import print_function
import numpy as np
import tensorflow as tf

import argparse
import time
import os
from six.moves import cPickle

from utils import TextLoader
from model import Model

from six import text_type

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='save',
                       help='model directory to store checkpointed models')
    parser.add_argument('-n', type=int, default=500,
                       help='number of characters to sample')
    parser.add_argument('--prime', type=text_type, default=u' ',
                       help='prime text')
    parser.add_argument('--sample', type=int, default=1,
                       help='0 to use max at each timestep, 1 to sample at each timestep, 2 to sample on spaces')

    args = parser.parse_args()
    sample(args)

def sample(args, init = True):
    global model
    global chars
    global vocab
    global saved_args
    if init:
        with open(os.path.join("save/last", 'config.pkl'), 'rb') as f:
            print('yo')
            saved_args = cPickle.load(f)
            with open(os.path.join("save/last", 'chars_vocab.pkl'), 'rb') as f:
                chars, vocab = cPickle.load(f)
                model = Model(saved_args, True)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(args['save_dir'])
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            return model.sample(sess, chars, vocab, args['n'], args['prime'], args['sample'])
            tf.reset_default_graph()

if __name__ == '__main__':
    main()
