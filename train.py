from __future__ import print_function
import numpy as np
import tensorflow as tf
import random

import argparse
import time
import os
from six.moves import cPickle

from utils import TextLoader
from model import Model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/python',
                       help='data directory containing scikit_cleaned.txt')
    parser.add_argument('--save_dir', type=str, default='save',
                       help='directory to store checkpointed models')
    parser.add_argument('--rnn_size', type=int, default=128,
                       help='size of RNN hidden state')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='number of layers in the RNN')
    parser.add_argument('--model', type=str, default='lstm',
                       help='rnn, gru, or lstm')
    parser.add_argument('--batch_size', type=int, default=50,
                       help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=50,
                       help='RNN sequence length')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='number of epochs')
    parser.add_argument('--save_every', type=int, default=1000,
                       help='save frequency')
    parser.add_argument('--grad_clip', type=float, default=5.,
                       help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.002,
                       help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.97,
                       help='decay rate for rmsprop')
    parser.add_argument('--init_from', type=str, default=None,
                       help="""continue training from saved model at this path. Path must contain files saved by previous training process:
                            'config.pkl'        : configuration;
                            'chars_vocab.pkl'   : vocabulary definitions;
                            'checkpoint'        : paths to model file(s) (created by tf).
                                                  Note: this file contains absolute paths, be careful when moving files around;
                            'model.ckpt-*'      : file(s) with model definition (created by tf)
                        """)
    args = parser.parse_args()
    train(args)

def train(args):
    data_loader = TextLoader(args.data_dir, args.batch_size, args.seq_length)
    args.vocab_size = data_loader.vocab_size

    # check compatibility if training is continued from previously saved model
    if args.init_from is not None:
        # check if all necessary files exist
        assert os.path.isdir(args.init_from)," %s must be a a path" % args.init_from
        assert os.path.isfile(os.path.join(args.init_from,"config.pkl")),"config.pkl file does not exist in path %s"%args.init_from
        assert os.path.isfile(os.path.join(args.init_from,"chars_vocab.pkl")),"chars_vocab.pkl.pkl file does not exist in path %s" % args.init_from
        ckpt = tf.train.get_checkpoint_state(args.init_from)
        assert ckpt,"No checkpoint found"
        assert ckpt.model_checkpoint_path,"No model path found in checkpoint"

        # open old config and check if models are compatible
        with open(os.path.join(args.init_from, 'config.pkl'), 'rb') as f:
            saved_model_args = cPickle.load(f)
        need_be_same=["model","rnn_size","num_layers","seq_length"]
        for checkme in need_be_same:
            assert vars(saved_model_args)[checkme]==vars(args)[checkme],"Command line argument and saved model disagree on '%s' "%checkme

        # open saved vocab/dict and check if vocabs/dicts are compatible
        with open(os.path.join(args.init_from, 'chars_vocab.pkl'), 'rb') as f:
            saved_chars, saved_vocab = cPickle.load(f)
        assert saved_chars==data_loader.chars, "Data and loaded model disagree on character set!"
        assert saved_vocab==data_loader.vocab, "Data and loaded model disagree on dictionary mappings!"
    try:
        with open(os.path.join(args.save_dir, 'config.pkl'), 'wb') as f:
            cPickle.dump(args, f)
        with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'wb') as f:
            cPickle.dump((data_loader.chars, data_loader.vocab), f)
    except FileNotFoundError as e:
        os.mkdir(args.save_dir)
        with open(os.path.join(args.save_dir, 'config.pkl'), 'wb') as f:
            cPickle.dump(args, f)
        with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'wb') as f:
            cPickle.dump((data_loader.chars, data_loader.vocab), f)


    model = Model(args)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        # restore model
        if args.init_from is not None:
            saver.restore(sess, ckpt.model_checkpoint_path)
        
        # creation validation batches
        num_validation_batches = int(data_loader.num_batches/10)
        validation_batches = list(set([random.randint(0,data_loader.num_batches) for i in range(num_validation_batches)]))
        num_validation_batches = len(validation_batches)
        validation_feed = {}
        validation_feed["x"] = []
        validation_feed["y"] = []

        print("creating validation batches...")
        for b in range(data_loader.num_batches):
            data_loader.reset_batch_pointer()
            x, y = data_loader.next_batch()
            if b in validation_batches:
                    validation_feed["x"].append(x)
                    validation_feed["y"].append(y)
            else:
                pass

        validation_loss_end = 0
        for e in range(args.num_epochs):
            sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** e)))
            data_loader.reset_batch_pointer()
            state = sess.run(model.initial_state)
            for b in range(data_loader.num_batches):
                start = time.time()
                x, y = data_loader.next_batch()
                feed = {model.input_data: x, model.targets: y}
                for i, (c, h) in enumerate(model.initial_state):
                    feed[c] = state[i].c
                    feed[h] = state[i].h

                train_loss, state, _ = sess.run([model.cost, model.final_state, model.train_op], feed)
                end = time.time()
                print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
                    .format(e * data_loader.num_batches + b,
                            args.num_epochs * data_loader.num_batches,
                            e, train_loss, end - start))

                if (e * data_loader.num_batches + b) % args.save_every == 0\
                    or (e==args.num_epochs-1 and b == data_loader.num_batches-1): # save for the last result
                    checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step = e * data_loader.num_batches + b)
                    print("model saved to {}".format(checkpoint_path))

                    # computation of the validation loss
                    validation_loss = 0
                    print("computing validation loss")
                    for i in range(num_validation_batches):
                        print("computing loss on the {}th validation batch".format(i))
                        validation_feed_batch = {}
                        validation_feed_batch[model.input_data] = validation_feed["x"][i]
                        validation_feed_batch[model.targets] = validation_feed["y"][i]
                        validation_loss += sess.run([model.cost, model.final_state], feed_dict=validation_feed_batch)[0]
                    validation_loss = validation_loss/num_validation_batches
                    print("validation loss=", validation_loss)
                    validation_loss_end = validation_loss

        print("validation loss after training =", validation_loss_end)

        return validation_loss_global
if __name__ == '__main__':
    main()
