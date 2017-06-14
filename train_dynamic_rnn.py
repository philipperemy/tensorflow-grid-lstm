import argparse
import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf

from data_dynamic_rnn import next_batch
from model_dynamic_rnn import Model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/tinyshakespeare',
                        help='data directory containing input.txt')
    parser.add_argument('--save_dir', type=str, default='save',
                        help='directory to store checkpointed models')
    parser.add_argument('--rnn_size', type=int, default=128,
                        help='size of RNN hidden state')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers in the RNN')
    parser.add_argument('--model', type=str, default='lstm',
                        help='rnn, gru, lstm, gridlstm, gridgru')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=50,
                        help='RNN sequence length')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='number of epochs')
    parser.add_argument('--save_every', type=int, default=100,
                        help='save frequency')
    parser.add_argument('--grad_clip', type=float, default=5.,
                        help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.002,
                        help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.97,
                        help='decay rate for rmsprop')
    args = parser.parse_args()
    print(args)
    train(args)


def train(args):
    args.num_bins = 5
    args.num_days = 10
    args.rnn_size = args.num_bins
    args.seq_length = args.num_days

    model = Model(args)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        train_loss_iterations = {'iteration': [], 'epoch': [], 'train_loss': [], 'val_loss': []}

        batch_idx = 0
        for e in range(args.num_epochs):
            sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** e)))
            state = sess.run(model.initial_state)
            for ba in range(int(1e9)):
                start = time.time()
                x, y = next_batch(args.batch_size, args.num_days, args.num_bins)

                b = np.hstack(
                    [np.mean(x, axis=1, keepdims=True)] +
                    [np.mean(x[:, 0:step, :], axis=1, keepdims=True) for step in
                     range(1, x.shape[1])])
                feed = {model.input_data: x, model.targets: y, model.initial_state: state}
                train_loss, state, _ = sess.run([model.cost, model.final_state, model.train_op], feed)
                end = time.time()
                batch_idx += 1

                benchmark_loss = np.mean(np.square(b - y))
                print("batch id #{} "
                      "train_loss = {:.3f}, "
                      "bench_loss = {:.3f}, "
                      "time/batch = {:.3f}".format(batch_idx, train_loss, benchmark_loss, end - start))
                train_loss_iterations['iteration'].append(batch_idx)
                train_loss_iterations['epoch'].append(e)
                train_loss_iterations['train_loss'].append(train_loss)

                if batch_idx % args.save_every == 0:

                    # evaluate
                    state_val = sess.run(model.initial_state)
                    x_val, y_val = next_batch(args.batch_size, args.num_days, args.num_bins)
                    feed_val = {model.input_data: x_val, model.targets: y_val, model.initial_state: state_val}
                    val_loss, state_val, _ = sess.run([model.cost, model.final_state, model.train_op], feed_val)
                    print('val_loss: {:.3f}'.format(val_loss))
                    train_loss_iterations['val_loss'].append(val_loss)

                    checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=batch_idx)
                    print("model saved to {}".format(checkpoint_path))
                else:
                    train_loss_iterations['val_loss'].append(None)

            pd.DataFrame(data=train_loss_iterations,
                         columns=train_loss_iterations.keys()).to_csv(os.path.join(args.save_dir, 'log.csv'))


if __name__ == '__main__':
    main()
