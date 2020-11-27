from ..summary import *
from lib.layers.basic import *
from lib.train.problem import Problem


def word_dropout(inp, inp_len, dropout, method, voc):
    inp_shape = tf.shape(inp)

    border = tf.fill([inp_shape[0], 1], False)

    mask = tf.sequence_mask(inp_len - 2, inp_shape[1] - 2)
    mask = tf.concat((border, mask, border), axis=1)
    mask = tf.logical_and(mask, tf.random_uniform(inp_shape) < dropout)

    if method == 'unk':
        replacement = tf.fill(inp_shape, tf.cast(voc._unk, inp.dtype))
    elif method == 'random_word':
        replacement = tf.random_uniform(inp_shape, minval=max(voc.bos, voc.eos, voc._unk)+1, maxval=voc.size(), dtype=inp.dtype)
    else:
        raise ValueError("Unknown word dropout method: %r" % method)

    return tf.where(mask, replacement, inp)


class DefaultProblem(Problem):

    def __init__(self, models, sum_loss=False, use_small_batch_multiplier=False,
        inp_word_dropout=0, out_word_dropout=0, word_dropout_method='unk',
    ):
        assert len(models) == 1

        self.models = models
        self.model = list(self.models.values())[0]

        self.inp_voc = self.model.inp_voc
        self.out_voc = self.model.out_voc

        self.sum_loss = sum_loss
        self.use_small_batch_multiplier = use_small_batch_multiplier

        self.inp_word_dropout = inp_word_dropout
        self.out_word_dropout = out_word_dropout
        self.word_dropout_method = word_dropout_method

        if self.use_small_batch_multiplier:
            self.max_batch_size_var = tf.get_variable("max_batch_size", shape=[], initializer=tf.ones_initializer(), trainable=False)

    def _make_encdec_batch(self, batch, is_train):
        encdec_batch = copy(batch)

        if is_train and self.inp_word_dropout > 0:
            encdec_batch['inp'] = word_dropout(encdec_batch['inp'], encdec_batch['inp_len'], self.inp_word_dropout, self.word_dropout_method, self.model.inp_voc)

        if is_train and self.out_word_dropout > 0:
            encdec_batch['out'] = word_dropout(encdec_batch['out'], encdec_batch['out_len'], self.out_word_dropout, self.word_dropout_method, self.model.out_voc)

        return encdec_batch

    def batch_counters(self, batch, is_train):
        if hasattr(self.model, 'batch_counters'):
            return self.model.batch_counters(batch, is_train)

        rdo = self.model.encode_decode(self._make_encdec_batch(batch, is_train), is_train)

        with dropout_scope(is_train):
            logits = self.model.loss.rdo_to_logits(rdo, batch['out'], batch['out_len'])  # [batch_size * nout * ovoc_size]
            loss_values = self.model.loss.logits2loss(logits, batch['out'], batch['out_len'])

        counters = dict(
            loss=tf.reduce_sum(loss_values),
            out_len=tf.to_float(tf.reduce_sum(batch['out_len'])),
        )
        append_counters_common_metrics(counters, logits, batch['out'], batch['out_len'], is_train)
        append_counters_xent(counters, loss_values, batch['out_len'])
        append_counters_io(counters, batch['inp'], batch['out'], batch['inp_len'], batch['out_len'])
        return counters

    def get_xent(self, batch, is_train):
        if hasattr(self.model, 'batch_counters'):
            return self.model.batch_counters(batch, is_train)

        rdo = self.model.encode_decode(self._make_encdec_batch(batch, is_train), is_train)

        with dropout_scope(is_train):
            logits = self.model.loss.rdo_to_logits(rdo, batch['out'],
                                                   batch['out_len'])  # [batch_size * nout * ovoc_size]
            loss_values = self.model.loss.logits2loss(logits, batch['out'], batch['out_len'])

        return loss_values

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

        return value

    def summary_multibatch(self, counters, prefix, is_train):
        res = []
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
