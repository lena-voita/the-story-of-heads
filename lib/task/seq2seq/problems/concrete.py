
from ..summary import *
from lib.layers.basic import *
from lib.train.problem import Problem
from lib.task.seq2seq.problems.default import word_dropout


class ConcreteProblem(Problem):
    def __init__(self, models, dump_dir=None, dump_first_n=None, sum_loss=False, use_small_batch_multiplier=False,
                 inp_word_dropout=0, out_word_dropout=0, word_dropout_method='unk', concrete_coef=1.,
                 ):
        assert len(models) == 1

        self.models = models
        self.model = list(self.models.values())[0]

        self.inp_voc = self.model.inp_voc
        self.out_voc = self.model.out_voc

        self.dump_dir = dump_dir
        self.dump_first_n = dump_first_n
        self.sum_loss = sum_loss
        self.use_small_batch_multiplier = use_small_batch_multiplier

        self.inp_word_dropout = inp_word_dropout
        self.out_word_dropout = out_word_dropout
        self.word_dropout_method = word_dropout_method

        # ========================  for concrete gates  =========================================
        self.concrete_coef = concrete_coef
        # ========================================================================================

        if self.use_small_batch_multiplier:
            self.max_batch_size_var = tf.get_variable("max_batch_size", shape=[], initializer=tf.ones_initializer(),
                                                      trainable=False)

    def _make_encdec_batch(self, batch, is_train):
        encdec_batch = copy(batch)

        if is_train and self.inp_word_dropout > 0:
            encdec_batch['inp'] = word_dropout(encdec_batch['inp'], encdec_batch['inp_len'], self.inp_word_dropout,
                                               self.word_dropout_method, self.model.inp_voc)

        if is_train and self.out_word_dropout > 0:
            encdec_batch['out'] = word_dropout(encdec_batch['out'], encdec_batch['out_len'], self.out_word_dropout,
                                               self.word_dropout_method, self.model.out_voc)

        return encdec_batch

    def batch_counters(self, batch, is_train):
        if hasattr(self.model, 'batch_counters'):
            return self.model.batch_counters(batch, is_train)

        # ========================  for concrete gates  =========================================
        tf.get_default_graph().clear_collection("CONCRETE")
        tf.get_default_graph().clear_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

        rdo = self.model.encode_decode(self._make_encdec_batch(batch, is_train), is_train)

        sparsity_rate = tf.reduce_mean(tf.get_collection("CONCRETE"))
        concrete_reg = tf.reduce_mean(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        # ========================================================================================

        with lib.layers.basic.dropout_scope(is_train):
            logits = self.model.loss.rdo_to_logits(rdo, batch['out'],
                                                   batch['out_len'])  # [batch_size * nout * ovoc_size]
            loss_values = self.model.loss.logits2loss(logits, batch['out'], batch['out_len'])
            # loss_values /= math.log(2.0)  # TODO: move to loss or to model

        if self.dump_dir:
            dump_map = batch

            loss_values = tf_dump(
                loss_values,
                dump_map,
                self.dump_dir + '/batch_dump_{}.npz',
                first_n=self.dump_first_n)

        counters = dict(
            loss=tf.reduce_sum(loss_values),
            out_len=tf.to_float(tf.reduce_sum(batch['out_len'])),
            # ========================  for concrete gates  =========================================
            sparsity_rate=sparsity_rate,
            concrete_reg=concrete_reg,
            # ========================================================================================
        )
        append_counters_common_metrics(counters, logits, batch['out'], batch['out_len'], is_train)
        append_counters_xent(counters, loss_values, batch['out_len'])
        append_counters_io(counters, batch['inp'], batch['out'], batch['inp_len'], batch['out_len'])
        return counters

    def loss_multibatch(self, counters, is_train):
        if self.sum_loss:
            value = tf.reduce_sum(counters['loss'])
        else:
            value = tf.reduce_sum(counters['loss']) / tf.reduce_sum(counters['out_len'])

        if self.use_small_batch_multiplier and is_train:
            batch_size = tf.reduce_sum(counters['out_len'])
            max_batch_size = tf.maximum(self.max_batch_size_var, batch_size)
            with tf.control_dependencies([tf.assign(self.max_batch_size_var, max_batch_size)]):
                small_batch_multiplier = batch_size / max_batch_size
                value = value * small_batch_multiplier

        # ========================  for concrete gates  =========================================
        value += self.concrete_coef * tf.reduce_mean(counters['concrete_reg'])
        # ========================================================================================

        return value

    def summary_multibatch(self, counters, prefix, is_train):
        res = []
        # ========================  for concrete gates  =========================================
        res.append(tf.summary.scalar(prefix + "/concrete_reg", tf.reduce_mean(counters['concrete_reg'])))
        res.append(tf.summary.scalar(prefix + "/sparsity_rate", tf.reduce_mean(counters['sparsity_rate'])))
        # ========================================================================================

        res += summarize_common_metrics(counters, prefix)
        res += summarize_xent(counters, prefix)
        res += summarize_io(counters, prefix)
        return res

    def params_summary(self):
        if hasattr(self.model, 'params_summary'):
            return self.model.params_summary()
        return []

    def make_feed_dict(self, batch, **kwargs):
        return self.model.make_feed_dict(batch, **kwargs)


