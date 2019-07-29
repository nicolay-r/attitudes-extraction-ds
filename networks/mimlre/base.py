import tensorflow as tf

from networks.context.architectures.base import BaseContextNeuralNetwork
from networks.context.architectures.utils import get_two_layer_logits
from networks.context.processing.sample import Sample
from networks.io import DataType
from networks.mimlre.configuration.base import MIMLRESettings
from networks.mimlre.processing.batch import MultiInstanceBatch
from networks.network import NeuralNetwork


class MIMLRE(NeuralNetwork):

    _context_network_scope_name = "context_network"

    # TODO. remove config
    def __init__(self, context_network):
        assert(isinstance(context_network, BaseContextNeuralNetwork))
        self.context_network = context_network
        self.cfg = None

    @property
    def ContextsPerOpinion(self):
        return self.cfg.BagSize

    def compile(self, config, reset_graph):
        assert(isinstance(config, MIMLRESettings))

        self.cfg = config
        tf.reset_default_graph()

        with tf.variable_scope("context-network"):
            self.context_network.compile(config=config.ContextSettings, reset_graph=False)

        self.init_input()
        self.init_hidden_states()

        max_pooling = self.init_body()
        logits_unscaled, logits_unscaled_dropped = get_two_layer_logits(
            max_pooling,
            self.W1, self.b1,
            self.W2, self.b2,
            self.dropout_keep_prob,
            activations=[tf.tanh, tf.tanh, None])
        output = tf.nn.softmax(logits_unscaled)
        self.labels = tf.cast(tf.argmax(output, axis=1), tf.int32)

        with tf.name_scope("cost"):
            self.weights, self.cost = BaseContextNeuralNetwork.init_weighted_cost(
                logits_unscaled_dropout=logits_unscaled_dropped,
                true_labels=self.y,
                config=config)

        with tf.name_scope("accuracy"):
            self.accuracy = BaseContextNeuralNetwork.init_accuracy(labels=self.labels, true_labels=self.y)

    def init_body(self):
        assert(isinstance(self.cfg, MIMLRESettings))

        with tf.name_scope("mimlre-body"):

            def __process_opinion(i, opinions, opinion_lens, results):
                """
                i, *, *
                """
                opinion = tf.gather(opinions, [i], axis=0)              # [1, sentences, embedding]
                opinion = tf.squeeze(opinion, [0])                      # [sentences, embedding]
                opinion_len = tf.gather(opinion_lens, [i], axis=0)      # [len]

                opinion = tf.reshape(opinion, [self.ContextsPerOpinion, self.context_network.ContextEmbeddingSize])

                slice_size = tf.pad(opinion_len, [[0, 1]], constant_values=self.__get_context_embedding_size(opinion))
                slice_size = tf.cast(slice_size, dtype=tf.int32)

                opinion = tf.slice(opinion, [0, 0], slice_size)

                pad_len = self.ContextsPerOpinion - opinion_len         # [s-len]
                pad_len = tf.pad(pad_len, [[1, 2]])                     # [0, s-len, 0, 0]
                pad_len = tf.reshape(pad_len, [2, 2])                   # [[0, s-len] [0, 0]]
                pad_len = tf.cast(pad_len, dtype=tf.int32)

                result = tf.pad(tensor=opinion,
                                paddings=pad_len,
                                constant_values=-1)                     # [s, embedding]
                outputs = results.write(i, result)

                i += 1
                return (i, opinions, opinion_lens, outputs)

            def __process_context(i, context_embeddings):
                """
                *, i, *
                Context handler.
                """
                def __to_ctx(tensor):
                    """
                    v: [batch, contexts, embedding] -> [batch, embedding]
                    """
                    return tf.squeeze(tf.gather(tensor, [i], axis=1), [1])

                self.context_network.x = __to_ctx(self.x)
                self.context_network.dist_from_subj = __to_ctx(self.dist_from_subj)
                self.context_network.dist_from_obj = __to_ctx(self.dist_from_obj)
                self.context_network.term_type = __to_ctx(self.term_type)
                self.context_network.pos = __to_ctx(self.pos)
                self.context_network.p_subj_ind = __to_ctx(self.p_subj_ind)
                self.context_network.p_obj_ind = __to_ctx(self.p_obj_ind)
                self.context_network.dropout_keep_prob = self.dropout_keep_prob
                self.context_network.embedding_dropout_keep_prob = self.embedding_dropout_keep_prob

                embedded_terms = self.context_network.init_embedded_input()
                context_embedding = self.context_network.init_context_embedding(embedded_terms)

                return i + 1, context_embeddings.write(i, context_embedding)

            def __condition_process_contexts(i, context_embeddings):
                return i < self.ContextsPerOpinion

            def __iter_x_by_contexts(handler):
                context_embeddings_arr = tf.TensorArray(
                    dtype=tf.float32,
                    name="contexts_arr",
                    size=self.ContextsPerOpinion,
                    infer_shape=False,
                    dynamic_size=True)

                _, context_embeddings = tf.while_loop(
                    __condition_process_contexts,
                    handler,
                    [0, context_embeddings_arr])

                return context_embeddings.stack()

            def __iter_x_by_opinions(x, handler, opinion_lens):
                """
                x:            [batch_size, sentences, embedding]
                opinion_lens: [batch_size, len]
                """
                context_iter = tf.TensorArray(
                    dtype=tf.float32,
                    name="context_iter",
                    size=self.cfg.BatchSize,
                    infer_shape=False,
                    dynamic_size=True)

                _, _, _, output = tf.while_loop(
                    lambda i, *_: tf.less(i, self.cfg.BatchSize),
                    handler,
                    (0, x, opinion_lens, context_iter))

                return output.stack()

            """
            Body
            """
            context_outputs = __iter_x_by_contexts(__process_context)                    # [sentences, batches, embedding]
            context_outputs = tf.transpose(context_outputs, perm=[1, 0, 2])              # [batches, sentences, embedding]
            sliced_contexts = __iter_x_by_opinions(
                x=context_outputs,
                handler=__process_opinion,
                opinion_lens=self.__calculate_opinion_lens(self.x))

            return self.__contexts_max_pooling(context_outputs=sliced_contexts,
                                               contexts_per_opinion=self.ContextsPerOpinion)  # [batches, max_pooling]

    @staticmethod
    def __get_context_embedding_size(opinion):
        return opinion.get_shape().as_list()[-1]

    def init_hidden_states(self):
        self.W1 = tf.Variable(tf.random_normal([self.context_network.ContextEmbeddingSize, self.cfg.HiddenSize]), dtype=tf.float32)
        self.W2 = tf.Variable(tf.random_normal([self.cfg.HiddenSize, self.cfg.ClassesCount]), dtype=tf.float32)
        self.b1 = tf.Variable(tf.random_normal([self.cfg.HiddenSize]), dtype=tf.float32)
        self.b2 = tf.Variable(tf.random_normal([self.cfg.ClassesCount]), dtype=tf.float32)

    @staticmethod
    def __calculate_opinion_lens(x):
        relevant = tf.sign(tf.abs(x))
        reduced_sentences = tf.reduce_max(relevant, reduction_indices=-1)
        length = tf.reduce_sum(reduced_sentences, reduction_indices=-1)
        length = tf.cast(length, tf.int64)
        return length

    @staticmethod
    def __contexts_max_pooling(context_outputs, contexts_per_opinion):
        context_outputs = tf.expand_dims(context_outputs, 0)     # [1, batches, sentences, embedding]
        context_outputs = tf.nn.max_pool(
            context_outputs,
            ksize=[1, 1, contexts_per_opinion, 1],
            strides=[1, 1, contexts_per_opinion, 1],
            padding='VALID',
            data_format="NHWC")
        return tf.squeeze(context_outputs)                       # [batches, max_pooling]

    def init_input(self):
        """
        These parameters actually are the same as in context model, but the shape has
        contexts count -- extra parameter.
        """
        contexts_count = self.cfg.BagSize
        batch_size = self.cfg.BagsPerMinibatch

        self.x = tf.placeholder(dtype=tf.int32, shape=[batch_size, contexts_count, self.cfg.TermsPerContext])
        self.y = tf.placeholder(dtype=tf.int32, shape=[batch_size])

        self.dist_from_subj = tf.placeholder(dtype=tf.int32, shape=[batch_size, contexts_count, self.cfg.TermsPerContext])
        self.dist_from_obj = tf.placeholder(dtype=tf.int32, shape=[batch_size, contexts_count,  self.cfg.TermsPerContext])
        self.term_type = tf.placeholder(tf.float32, shape=[batch_size, contexts_count, self.cfg.TermsPerContext])
        self.pos = tf.placeholder(tf.int32, shape=[batch_size, contexts_count, self.cfg.TermsPerContext])
        self.p_subj_ind = tf.placeholder(dtype=tf.int32, shape=[batch_size, contexts_count])
        self.p_obj_ind = tf.placeholder(dtype=tf.int32, shape=[batch_size, contexts_count])

        self.dropout_keep_prob = tf.placeholder(tf.float32)
        self.embedding_dropout_keep_prob = tf.placeholder(tf.float32)

    def create_feed_dict(self, input, data_type):

        feed_dict = {
            self.x: input[Sample.I_X_INDS],
            self.y: input[MultiInstanceBatch.I_LABELS],
            self.dist_from_subj: input[Sample.I_SUBJ_DISTS],
            self.dist_from_obj: input[Sample.I_OBJ_DISTS],
            self.term_type: input[Sample.I_TERM_TYPE],
            self.p_subj_ind: input[Sample.I_SUBJ_IND],
            self.p_obj_ind: input[Sample.I_OBJ_IND],
            self.dropout_keep_prob: self.cfg.DropoutKeepProb if data_type == DataType.Train else 1.0,
            self.embedding_dropout_keep_prob: self.cfg.EmbeddingDropoutKeepProb if data_type == DataType.Train else 1.0
        }

        if self.cfg.UsePOSEmbedding:
            feed_dict[self.pos] = input[Sample.I_POS_INDS]

        return feed_dict

    @property
    def Labels(self):
        return self.labels

    @property
    def Accuracy(self):
        return self.accuracy

    @property
    def Cost(self):
        return self.cost

    @property
    def ParametersDictionary(self):
        return {}

    @property
    def Variables(self):
        return [], []

