import tensorflow as tf
from tensorflow.contrib import grid_rnn
from tensorflow.contrib import rnn as rnn_cell


class Model(object):
    def __init__(self, args, infer=False):
        self.args = args
        if infer:
            args.batch_size = 1
            args.seq_length = 1

        additional_cell_args = {}
        if args.model == 'rnn':
            cell_fn = rnn_cell.BasicRNNCell
        elif args.model == 'gru':
            cell_fn = rnn_cell.GRUCell
        elif args.model == 'lstm':
            cell_fn = rnn_cell.BasicLSTMCell
        elif args.model == 'gridlstm':
            cell_fn = grid_rnn.Grid2LSTMCell
            additional_cell_args.update({'use_peepholes': True, 'forget_bias': 1.0})
        elif args.model == 'gridgru':
            cell_fn = grid_rnn.Grid2GRUCell
        else:
            raise Exception("model type not supported: {}".format(args.model))

        cell = cell_fn(args.rnn_size, **additional_cell_args)

        self.cell = cell = rnn_cell.MultiRNNCell([cell] * args.num_layers)

        self.input_data = tf.placeholder(tf.float32, [args.batch_size, args.seq_length, args.rnn_size])
        self.targets = tf.placeholder(tf.float32, [args.batch_size, args.seq_length, args.rnn_size])
        self.initial_state = cell.zero_state(args.batch_size, tf.float32)

        from tensorflow.python.ops.rnn import dynamic_rnn
        outputs, last_state = dynamic_rnn(self.cell,
                                          self.input_data,
                                          initial_state=self.initial_state,
                                          dtype=tf.float32)
        self.cost = tf.reduce_mean(tf.squared_difference(outputs, self.targets))
        self.final_state = last_state

        self.lr = tf.Variable(0.0, trainable=False)
        t_vars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, t_vars), args.grad_clip)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, t_vars))
