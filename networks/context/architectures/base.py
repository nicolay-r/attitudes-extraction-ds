import tensorflow as tf

from networks.context.architectures.attention.base import Attention
from networks.context.configurations.base import CommonModelSettings
from networks.context.processing.batch import MiniBatch
from networks.context.processing.sample import Sample
from networks.io import DataType
from networks.network import NeuralNetwork


class BaseContextNeuralNetwork(NeuralNetwork):

    _attention_var_scope_name = 'attention-model'

    def __init__(self):
        self.cfg = None
        self.__att_weights = None
        self.__labels = None

    def compile(self, config, reset_graph):
        assert(isinstance(config, CommonModelSettings))
        assert(isinstance(reset_graph, bool))

        self.cfg = config

        if reset_graph:
            tf.reset_default_graph()

        if self.cfg.UseAttention:
            with tf.variable_scope(self._attention_var_scope_name):
                self.cfg.AttentionModel.init_hidden()

        self.init_input()
        self.init_embedding_hidden_states()
        self.init_hidden_states()

        embedded_terms = self.init_embedded_input()
        context_embedding = self.init_context_embedding(embedded_terms)
        logits_unscaled, logits_unscaled_dropped = self.init_logits_unscaled(context_embedding)

        # Get output for each sample
        output = tf.nn.softmax(logits_unscaled)
        # Create labels only for whole bags
        self.__labels = tf.cast(tf.argmax(self.to_mean_of_bag(output), axis=1), tf.int32)

        with tf.name_scope("cost"):
            self.weights, self.cost = self.init_weighted_cost(
                logits_unscaled_dropout=self.to_mean_of_bag(logits_unscaled_dropped),
                true_labels=self.y,
                config=config)

        with tf.name_scope("accuracy"):
            self.accuracy = self.init_accuracy(labels=self.Labels, true_labels=self.y)

    @property
    def ContextEmbeddingSize(self):
        raise Exception("Not implemented")

    def init_embedding_hidden_states(self):
        self.term_emb = tf.constant(value=self.cfg.TermEmbeddingMatrix,
                                    dtype=tf.float32,
                                    shape=self.cfg.TermEmbeddingShape)

        self.dist_emb = tf.get_variable(dtype=tf.float32,
                                        initializer=tf.random_normal_initializer,
                                        shape=[self.cfg.TermsPerContext, self.cfg.DistanceEmbeddingSize],
                                        trainable=True,
                                        name="dist_emb")

        self.pos_emb = tf.get_variable(dtype=tf.float32,
                                       initializer=tf.random_normal_initializer,
                                       shape=[len(self.cfg.PosTagger.pos_names), self.cfg.PosEmbeddingSize],
                                       trainable=True,
                                       name="pos_emb")

    def init_hidden_states(self):
        raise Exception("Not implemented")

    def init_context_embedding(self, embedded_terms):
        raise Exception("Not implemented")

    def init_logits_unscaled(self, context_embedding):
        raise Exception("Not implemented")

    def init_attention_embedding(self):
        assert(isinstance(self.cfg.AttentionModel, Attention))
        self.cfg.AttentionModel.set_x(self.x)
        self.cfg.AttentionModel.set_entities(tf.stack([self.p_subj_ind, self.p_obj_ind], axis=-1))         # [batch_size, 2]
        att_e, self.__att_weights = self.cfg.AttentionModel.init_body(self.term_emb)
        return att_e

    def init_embedded_input(self):
        return self.optional_process_embedded_data(self.cfg,
                                                   self.init_embedded_terms(),
                                                   self.embedding_dropout_keep_prob)

    @staticmethod
    def optional_process_embedded_data(config, embedded, dropout_keep_prob):
        assert(isinstance(config, CommonModelSettings))

        if config.UseEmbeddingDropout:
            return tf.nn.dropout(embedded, keep_prob=dropout_keep_prob)

        return embedded

    def init_input(self):
        """
        Input placeholders
        """
        self.x = tf.placeholder(dtype=tf.int32, shape=[self.cfg.BatchSize, self.cfg.TermsPerContext], name="ctx_x")
        self.y = tf.placeholder(dtype=tf.int32, shape=[self.cfg.BagsPerMinibatch], name="ctx_y")
        self.dist_from_subj = tf.placeholder(dtype=tf.int32, shape=[self.cfg.BatchSize, self.cfg.TermsPerContext], name="ctx_dist_from_subj")
        self.dist_from_obj = tf.placeholder(dtype=tf.int32, shape=[self.cfg.BatchSize, self.cfg.TermsPerContext], name="ctx_dist_from_obj")
        self.term_type = tf.placeholder(tf.float32, shape=[self.cfg.BatchSize, self.cfg.TermsPerContext], name="ctx_term_type")
        self.pos = tf.placeholder(tf.int32, shape=[self.cfg.BatchSize, self.cfg.TermsPerContext], name="ctx_pos")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="ctx_dropout_keep_prob")
        self.embedding_dropout_keep_prob = tf.placeholder(tf.float32, name="cxt_emb_dropout_keep_prob")
        self.p_subj_ind = tf.placeholder(dtype=tf.int32, shape=[self.cfg.BatchSize], name="ctx_p_subj_ind")
        self.p_obj_ind = tf.placeholder(dtype=tf.int32, shape=[self.cfg.BatchSize], name="ctx_p_obj_ind")

        if self.cfg.UseAttention:
            with tf.variable_scope(self._attention_var_scope_name):
                self.cfg.AttentionModel.init_input()

    @staticmethod
    def _get_attention_vector_size(cfg):
        return 0 if not cfg.UseAttention else cfg.AttentionModel.AttentionEmbeddingSize

    @property
    def TermEmbeddingSize(self):
        size = self.cfg.TermEmbeddingShape[1] + 2 * self.cfg.DistanceEmbeddingSize + 1

        if self.cfg.UsePOSEmbedding:
            size += self.cfg.PosEmbeddingSize

        return size

    def init_embedded_terms(self):

        embedded_terms = tf.concat([tf.nn.embedding_lookup(self.term_emb, self.x),
                                    tf.nn.embedding_lookup(self.dist_emb, self.dist_from_subj),
                                    tf.nn.embedding_lookup(self.dist_emb, self.dist_from_obj),
                                    tf.reshape(self.term_type, [self.cfg.BatchSize, self.cfg.TermsPerContext, 1])],
                                   axis=-1)

        if self.cfg.UsePOSEmbedding:
            embedded_terms = tf.concat([embedded_terms,
                                        tf.nn.embedding_lookup(self.pos_emb, self.pos)],
                                       axis=-1)

        return embedded_terms

    @staticmethod
    def init_accuracy(labels, true_labels):
        correct = tf.equal(labels, true_labels)
        return tf.reduce_mean(tf.cast(correct, tf.float32))

    @staticmethod
    def init_weighted_cost(logits_unscaled_dropout, true_labels, config):
        """
        Init loss with weights for tensorflow model.
        'labels' suppose to be a list of indices (not priorities)
        """
        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits_unscaled_dropout,
            labels=true_labels)

        weights = tf.reduce_sum(
            config.ClassWeights * tf.one_hot(indices=true_labels, depth=config.ClassesCount),
            axis=1)

        if config.UseClassWeights:
            cost = cost * weights

        return weights, cost

    def to_mean_of_bag(self, logits):
        loss = tf.reshape(logits, [self.cfg.BagsPerMinibatch, self.cfg.BagSize, self.cfg.ClassesCount])
        return tf.reduce_mean(loss, axis=1)

    def create_feed_dict(self, input, data_type):

        feed_dict = {
            self.x: input[Sample.I_X_INDS],
            self.y: input[MiniBatch.I_LABELS],
            self.dist_from_subj: input[Sample.I_SUBJ_DISTS],
            self.dist_from_obj: input[Sample.I_OBJ_DISTS],
            self.term_type: input[Sample.I_TERM_TYPE],
            self.dropout_keep_prob: self.cfg.DropoutKeepProb if data_type == DataType.Train else 1.0,
            self.embedding_dropout_keep_prob: self.cfg.EmbeddingDropoutKeepProb if data_type == DataType.Train else 1.0,
            self.p_subj_ind: input[Sample.I_SUBJ_IND],
            self.p_obj_ind: input[Sample.I_OBJ_IND]
        }

        if self.cfg.UsePOSEmbedding:
            feed_dict[self.pos] = input[Sample.I_POS_INDS]

        return feed_dict

    @property
    def Labels(self):
        return self.__labels

    @property
    def Accuracy(self):
        return self.accuracy

    @property
    def Cost(self):
        return self.cost

    @property
    def Variables(self):
        assert(isinstance(self.cfg, CommonModelSettings))
        params = []

        if self.__att_weights is not None:
            params.append((ContextNetworkVariableNames.AttentionWeights, self.__att_weights))

        if len(params) == 0:
            return [], []

        return [list(p) for p in zip(*params)]


class ContextNetworkVariableNames(object):

    AttentionWeights = 'attention-weights'
