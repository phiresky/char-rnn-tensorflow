#!/usr/bin/env python

from __future__ import print_function

import argparse
import os
from six.moves import cPickle
import sys
import json

from six import text_type


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--save_dir', type=str, default='save',
                    help='model directory to store checkpointed models')

args = parser.parse_args()

import tensorflow as tf
from model import Model


def sample(args):
    with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
        saved_args = cPickle.load(f)
    with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'rb') as f:
        chars, vocab = cPickle.load(f)
    model = Model(saved_args, training=False)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(args.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            for line in sys.stdin:
                try:
                    print(f"SAMPLE {line}", file=sys.stderr)
                    parsed = json.loads(line)
                    outp = model.sample(sess, chars, vocab, parsed["n"], parsed["prime"],
                                        parsed["mode"])

                    print(json.dumps({'result': outp}))
                except Exception as e:
                    print(f"error: {str(e)}", file=sys.stderr)


if __name__ == '__main__':
    sample(args)
