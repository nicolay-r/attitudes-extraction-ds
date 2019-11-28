import tensorflow as tf
from tensorflow.python.ops import math_ops

from networks.context.architectures.base import BaseContextNeuralNetwork
from networks.context.configurations.ian import IANConfig
from networks.context.processing.sample import Sample
import utils

# (C) Peiqin Lin
# original: https://github.com/lpq29743/IAN/blob/master/model.py


class IAN(BaseContextNeuralNetwork):

    @property
    def ContextEmbeddingSize(self):
        return self.cfg.HiddenSize * 2

    def init_input(self):
        super(IAN, self).init_input()
        assert(isinstance(self.cfg, IANConfig))
        self.aspects = tf.placeholder(tf.int32, shape=[self.cfg.BatchSize, self.cfg.MaxAspectLength])
        self.E_aspects = tf.get_variable(name="E_aspects",
                                         dtype=tf.float32,
                                         initializer=tf.random_normal_initializer,
                                         shape=self.cfg.AspectsEmbeddingShape,
                                         trainable=True)

    def init_hidden_states(self):
        assert(isinstance(self.cfg, IANConfig))

        with tf.name_scope('weights'):
            self.weights = {
                'aspect_score': tf.get_variable(
                    name='W_a',
                    shape=[self.cfg.HiddenSize, self.cfg.HiddenSize],
                    initializer=tf.random_uniform_initializer(-0.1, 0.1),
                    regularizer=tf.contrib.layers.l2_regularizer(self.cfg.L2Reg)
                ),
                'context_score': tf.get_variable(
                    name='W_c',
                    shape=[self.cfg.HiddenSize, self.cfg.HiddenSize],
                    initializer=tf.random_uniform_initializer(-0.1, 0.1),
                    regularizer=tf.contrib.layers.l2_regularizer(self.cfg.L2Reg)
                ),
                'softmax': tf.get_variable(
                    name='W_l',
                    shape=[self.ContextEmbeddingSize(), self.cfg.ClassesCount],
                    initializer=tf.random_uniform_initializer(-0.1, 0.1),
                    regularizer=tf.contrib.layers.l2_regularizer(self.cfg.L2Reg)
                ),
            }

        with tf.name_scope('biases'):
            self.biases = {
                'aspect_score': tf.get_variable(
                    name='B_a',
                    shape=[self.cfg.MaxAspectLength, 1],
                    initializer=tf.zeros_initializer(),
                    regularizer=tf.contrib.layers.l2_regularizer(self.cfg.L2Reg)
                ),
                'context_score': tf.get_variable(
                    name='B_c',
                    shape=[self.cfg.MaxContextLength, 1],
                    initializer=tf.zeros_initializer(),
                    regularizer=tf.contrib.layers.l2_regularizer(self.cfg.L2Reg)
                ),
                'softmax': tf.get_variable(
                    name='B_l',
                    shape=[self.cfg.ClassesCount],
                    initializer=tf.zeros_initializer(),
                    regularizer=tf.contrib.layers.l2_regularizer(self.cfg.L2Reg)
                ),
            }

    def init_embedded_input(self):
        super(IAN, self).init_embedded_input()

        aspect_inputs = tf.cast(tf.nn.embedding_lookup(self.E_aspects, self.aspects), tf.float32)
        self.aspect_inputs = self.optional_process_embedded_data(
            self.cfg,
            aspect_inputs,
            self.embedding_dropout_keep_prob)

    def init_context_embedding(self, embedded_terms):
        with tf.name_scope('inputs'):
            context_inputs = embedded_terms
            aspect_inputs = self.aspect_inputs

        with tf.name_scope('dynamic_rnn'):
            aspect_lens = utils.length(self.aspects)
            aspect_outputs, aspect_state = tf.nn.dynamic_rnn(
                tf.contrib.rnn.LSTMCell(self.cfg.HiddenSize),
                inputs=aspect_inputs,
                sequence_length=aspect_lens,
                dtype=tf.float32,
                scope='aspect_lstm'
            )
            aspect_avg = tf.reduce_mean(aspect_outputs, 1)

            context_lens = utils.length(self.x)
            context_outputs, context_state = tf.nn.dynamic_rnn(
                tf.contrib.rnn.LSTMCell(self.cfg.HiddenSize),
                inputs=context_inputs,
                sequence_length=context_lens,
                dtype=tf.float32,
                scope='context_lstm'
            )
            context_avg = tf.reduce_mean(context_outputs, 1)

            aspect_outputs_iter = tf.TensorArray(tf.float32, 1, dynamic_size=True, infer_shape=False)
            aspect_outputs_iter = aspect_outputs_iter.unstack(aspect_outputs)
            context_avg_iter = tf.TensorArray(tf.float32, 1, dynamic_size=True, infer_shape=False)
            context_avg_iter = context_avg_iter.unstack(context_avg)
            aspect_lens_iter = tf.TensorArray(tf.int64, 1, dynamic_size=True, infer_shape=False)
            aspect_lens_iter = aspect_lens_iter.unstack(aspect_lens)
            aspect_rep = tf.TensorArray(size=self.cfg.BatchSize, dtype=tf.float32)
            aspect_att = tf.TensorArray(size=self.cfg.BatchSize, dtype=tf.float32)

            def aspect_body(i, aspect_rep, aspect_att, weights, biases):
                a = aspect_outputs_iter.read(i)
                b = context_avg_iter.read(i)
                l = math_ops.to_int32(aspect_lens_iter.read(i))
                aspect_score = tf.reshape(tf.nn.tanh(
                    tf.matmul(tf.matmul(a,
                                        weights['aspect_score']),
                                        tf.reshape(b, [-1, 1])) + biases['aspect_score']),
                                        [1, -1])
                aspect_att_temp = tf.concat(
                    [tf.nn.softmax(tf.slice(aspect_score, [0, 0], [1, l])),
                     tf.zeros([1, self.cfg.MaxAspectLength - l])],
                     1)
                aspect_att = aspect_att.write(i, aspect_att_temp)
                aspect_rep = aspect_rep.write(i, tf.matmul(aspect_att_temp, a))
                return (i + 1, aspect_rep, aspect_att, weights, biases)

            def condition(i, aspect_rep, aspect_att, weights, biases):
                return i < self.cfg.BatchSize

            _, aspect_rep_final, aspect_att_final, _, _ = tf.while_loop(
                cond=condition,
                body=aspect_body,
                loop_vars=(0, aspect_rep, aspect_att, self.weights, self.biases))

            self.aspect_atts = tf.reshape(aspect_att_final.stack(), [-1, self.cfg.MaxAspectLength])
            self.aspect_reps = tf.reshape(aspect_rep_final.stack(), [-1, self.cfg.HiddenSize])

            context_outputs_iter = tf.TensorArray(tf.float32, 1, dynamic_size=True, infer_shape=False)
            context_outputs_iter = context_outputs_iter.unstack(context_outputs)
            aspect_avg_iter = tf.TensorArray(tf.float32, 1, dynamic_size=True, infer_shape=False)
            aspect_avg_iter = aspect_avg_iter.unstack(aspect_avg)
            context_lens_iter = tf.TensorArray(tf.int64, 1, dynamic_size=True, infer_shape=False)
            context_lens_iter = context_lens_iter.unstack(context_lens)
            context_rep = tf.TensorArray(size=self.cfg.BatchSize, dtype=tf.float32)
            context_att = tf.TensorArray(size=self.cfg.BatchSize, dtype=tf.float32)

            def context_body(i, context_rep, context_att, weights, biases):
                a = context_outputs_iter.read(i)
                b = aspect_avg_iter.read(i)
                l = math_ops.to_int32(context_lens_iter.read(i))
                context_score = tf.reshape(tf.nn.tanh(
                    tf.matmul(tf.matmul(a, self.weights['context_score']), tf.reshape(b, [-1, 1])) + self.biases['context_score']), [1, -1])
                context_att_temp = tf.concat(
                    [tf.nn.softmax(tf.slice(context_score, [0, 0], [1, l])), tf.zeros([1, self.cfg.MaxContextLength - l])],
                    1)
                context_att = context_att.write(i, context_att_temp)
                context_rep = context_rep.write(i, tf.matmul(context_att_temp, a))
                return (i + 1, context_rep, context_att, weights, biases)

            def condition(i, context_rep, context_att, weights, biases):
                return i < self.cfg.BatchSize

            _, context_rep_final, context_att_final, _, _ = tf.while_loop(
                cond=condition,
                body=context_body,
                loop_vars=(0, context_rep, context_att, self.weights, self.biases))

            self.context_atts = tf.reshape(context_att_final.stack(), [-1, self.cfg.MaxContextLength])
            self.context_reps = tf.reshape(context_rep_final.stack(), [-1, self.cfg.HiddenSize])

            return tf.concat([self.aspect_reps, self.context_reps], 1)

    def init_logits_unscaled(self, context_embedding):
        return utils.get_single_layer_logits(
            context_embedding,
            self.weights['softmax'], self.biases['softmax'],
            self.dropout_keep_prob)

    def create_feed_dict(self, input, data_type):
        feed_dict = super(IAN, self).create_feed_dict(input, data_type)

        subj_ind = input[Sample.I_SUBJ_IND]
        obj_ind = input[Sample.I_OBJ_IND]
        feed_dict[self.aspects] = [[subj_ind[i], obj_ind[i]] for i in range(len(input[Sample.I_X_INDS]))]
        return feed_dict

    @property
    def ParametersDictionary(self):
        return {}

    @property
    def Variables(self):
        return [], []