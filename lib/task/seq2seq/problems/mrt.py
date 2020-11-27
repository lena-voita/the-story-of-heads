import tensorflow as tf
import lib
from lib.task.seq2seq.summary import *
from copy import copy
import lib.layers.basic
from lib.layers.basic import infer_length
import nltk
import numpy as np

def pad_to_length(out, length, token):
    assert out.shape.ndims == 3
    num_paddings = tf.maximum(0, length - tf.shape(out)[2])
    padding = tf.fill([tf.shape(out)[0], tf.shape(out)[1], num_paddings], token)
    return tf.concat([out, padding], axis=2)


class PyBleuComputer:
    def __init__(self, voc, smoothing_function):
        self.voc, self.smoothing_function = voc, smoothing_function

    def crop_eos(self, seq):
        seq = list(seq)
        if self.voc.eos in seq:
            seq = seq[:seq.index(self.voc.eos)]
        return seq

    def py_compute_sentence_bleu(self, prediction, reference, debug=False):
        scores = []
        for pred_i, ref_i in zip(prediction, reference):
            pred_i, ref_i = map(self.crop_eos, [pred_i, ref_i])
            if len(pred_i) > 0 and len(ref_i) > 0:
                score_i = nltk.bleu([ref_i], pred_i,
                                    smoothing_function=self.smoothing_function)
            else:
                score_i = 0
            scores.append(score_i)
            if debug:
                print('pred and ref:', pred_i, ref_i)
                print('score:', score_i)
        return np.array(scores, dtype=np.float32)

    def __call__(self, prediction, reference):
        assert prediction.shape.ndims == 2 and reference.shape.ndims == 2
        bleu_scores = tf.py_func(self.py_compute_sentence_bleu, [prediction, reference],
                                 tf.float32, stateful=False, name='compute_sentence_bleu')
        bleu_scores.set_shape([None])
        return tf.stop_gradient(bleu_scores)


def word_dropout(inp, inp_len, dropout, method, voc):
    inp_shape = tf.shape(inp)

    border = tf.fill([inp_shape[0], 1], False)

    mask = tf.sequence_mask(inp_len - 2, inp_shape[1] - 2)
    mask = tf.concat((border, mask, border), axis=1)
    mask = tf.logical_and(mask, tf.random_uniform(inp_shape) < dropout)

    if method == 'unk':
        replacement = tf.fill(inp_shape, tf.cast(voc._unk, inp.dtype))
    elif method == 'random_word':
        replacement = tf.random_uniform(inp_shape, minval=max(voc.bos, voc.eos, voc._unk) + 1, maxval=voc.size(),
                                        dtype=inp.dtype)
    else:
        raise ValueError("Unknown word dropout method: %r" % method)

    return tf.where(mask, replacement, inp)


class MRTProblem(lib.train.Problem):
    """
        Minimum risk training as defined here: https://www.aclweb.org/anthology/P16-1159.pdf
    """

    def __init__(self, models, sum_loss=False, use_small_batch_multiplier=False,
                 inp_word_dropout=0, out_word_dropout=0, word_dropout_method='unk',
                 hypo_inference_flags={'mode': 'sample', 'sampling_strategy': 'random'},
                 num_hypos=100, alpha=5e-3, target_in_hypos=True, loss_type='minus_bleu',
                 mean_over_seq=False,
                 ):
        # loss_type: one of 'minus_bleu' or 'one_minus_bleu'
        assert len(models) == 1
        assert loss_type in ['minus_bleu', 'one_minus_bleu'], "Loss type has to be one of ['minus_bleu', 'one_minus_bleu']"

        self.models = models
        self.model = list(self.models.values())[0]

        self.inp_voc = self.model.inp_voc
        self.out_voc = self.model.out_voc

        self.sum_loss = sum_loss
        self.use_small_batch_multiplier = use_small_batch_multiplier

        self.inp_word_dropout = inp_word_dropout
        self.out_word_dropout = out_word_dropout
        self.word_dropout_method = word_dropout_method

        # ----- begin the MRT part -----
        self.hypo_inference_flags = hypo_inference_flags
        self.target_in_hypos = target_in_hypos
        self.loss_type = loss_type
        self.num_hypos = tf.constant(num_hypos)
        self.neg_bias_value = -1e9
        self.alpha = alpha
        self.get_max_len = lambda inp_len, out_len: tf.to_int32(tf.to_float(out_len) * 1.4) + 3
        self.mean_over_seq = mean_over_seq
        # ----- end the MRT part -----

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

    def concat_hypos_with_out(self, hypos, out, eos):
        assert out.shape.ndims == 2 and hypos.shape.ndims == 3
        # hypos shape: [batch_size, n_hypos, length1]
        # out shape: [batch_size, length2]

        out_reshaped = out[:, None, :]

        max_len_hypos = tf.shape(hypos)[2]
        max_len_out = tf.shape(out_reshaped)[2]
        new_max_len = tf.maximum(max_len_out, max_len_hypos)

        padded_hypos = pad_to_length(hypos, new_max_len, eos)
        padded_out = pad_to_length(out_reshaped, new_max_len, eos)
        return tf.concat([padded_out, padded_hypos], axis=1)

    def py_deduplicate_hypos(self, hypos_with_out):
        mask = np.full(hypos_with_out.shape[:-1], False, dtype=np.bool)

        for batch_i in range(hypos_with_out.shape[0]):
            all_hypos = list(map(tuple, hypos_with_out[batch_i]))
            unique_hypos = set(all_hypos)
            for hypo_i, hypo in enumerate(all_hypos):
                if hypo in unique_hypos:
                    unique_hypos.discard(hypo)
                    mask[batch_i, hypo_i] = True

        return mask

    def deduplicate_hypos(self, hypos_with_out):
        return tf.py_func(self.py_deduplicate_hypos, [hypos_with_out], tf.bool, stateful=False, name='deduplicate')

    def batch_counters(self, batch, is_train):

        wide_batch = copy(batch)
        wide_batch['inp'] = tf.expand_dims(wide_batch['inp'], axis=1)
        wide_batch['inp'] = tf.tile(wide_batch['inp'], [1, self.num_hypos, 1])
        wide_batch['inp'] = tf.reshape(wide_batch['inp'],
                                       [tf.shape(wide_batch['inp'])[0] * tf.shape(wide_batch['inp'])[1],
                                        tf.shape(wide_batch['inp'])[2]])

        new_out_ = tf.expand_dims(wide_batch['out'], axis=1)
        new_out_ = tf.tile(new_out_, [1, self.num_hypos + 1, 1])
        new_out_ = tf.reshape(new_out_, [-1, tf.shape(new_out_)[-1]])
        hypo_max_len = self.get_max_len(infer_length(wide_batch['inp'], self.inp_voc.eos),
                                        infer_length(new_out_, self.out_voc.eos))

        hypos = self.model.symbolic_translate(wide_batch, max_len=hypo_max_len, back_prop=False,
                                              **self.hypo_inference_flags).best_out
        # ========== add out to hypos ===========
        if self.target_in_hypos:
            hypos_per_inp = tf.reshape(hypos, [tf.shape(batch['inp'])[0], -1, tf.shape(hypos)[-1]])
            hypos_with_out = self.concat_hypos_with_out(hypos_per_inp, batch['out'], eos=self.out_voc.eos)
            # hypos_with_out: [batch_size, n_hypos, out_len]
            hypos_with_out_flat = tf.reshape(hypos_with_out, [-1, tf.shape(hypos_with_out)[-1]])
        else:
            hypos_with_out = tf.reshape(hypos, [tf.shape(batch['inp'])[0], -1, tf.shape(hypos)[-1]])
            hypos_with_out_flat = hypos

        # ========== prepare large batch with hypos and out ===========

        wide_batch_with_out = copy(batch)
        new_inp = tf.expand_dims(wide_batch_with_out['inp'], axis=1)
        new_inp = tf.tile(new_inp, [1, self.num_hypos + (1 if self.target_in_hypos else 0), 1])
        new_inp = tf.reshape(new_inp, [-1, tf.shape(new_inp)[-1]])

        new_out = tf.expand_dims(wide_batch_with_out['out'], axis=1)
        new_out = tf.tile(new_out, [1, self.num_hypos + (1 if self.target_in_hypos else 0), 1])
        new_out = tf.reshape(new_out, [-1, tf.shape(new_out)[-1]])

        new_inp_len = lib.ops.basic.infer_length(new_inp, eos=self.inp_voc.eos)
        new_out_len = lib.ops.basic.infer_length(hypos_with_out_flat, eos=self.out_voc.eos)

        # ========== mask for duplicates ===========
        duplicate_mask = self.deduplicate_hypos(hypos_with_out)  # duplicate_mask: [batch_size, n_hypos]
        duplicate_mask_flat = tf.reshape(duplicate_mask, tf.shape(new_out_len))  # [batch_size * n_hypos]

        # ========== eval logprobs for all hypos ===========
        new_batch = {'inp': new_inp, 'out': hypos_with_out_flat,
                     'inp_len': new_inp_len, 'out_len': new_out_len}

        rdo = self.model.encode_decode(new_batch, is_train=is_train)
        with lib.layers.basic.dropout_scope(is_train):
            logits = self.model.loss.rdo_to_logits(rdo, new_batch['out'],
                                                   new_batch['out_len'])  # [batch_size * nout * ovoc_size]
            sent_logprobs = - self.model.loss.logits2loss(logits, new_batch['out'], new_batch['out_len'])
            if self.mean_over_seq:
                sent_logprobs /= tf.to_float(new_batch['out_len'])

        sent_logprobs *= self.alpha
        sent_logprobs += self.neg_bias_value * (1 - tf.to_float(duplicate_mask_flat))  # batch_size * (num_hypos + 1)

        sent_logprobs = tf.reshape(sent_logprobs, [tf.shape(batch['inp'])[0], -1])

        sent_probs = tf.nn.softmax(sent_logprobs, axis=-1)
        sent_probs_flat = tf.reshape(sent_probs, tf.shape(new_out_len))

        # ========== eval bleu for all hypos ===========
        compute_bleu = PyBleuComputer(self.model.out_voc, nltk.bleu_score.SmoothingFunction().method1)
        bleu_scores_flat = compute_bleu(hypos_with_out_flat, new_out)

        # ========== final_loss ===========
        if self.loss_type == 'minus_bleu':
            loss_values = - sent_probs_flat * bleu_scores_flat  # [batch_size * (num_hypos + 1)]
        else:
            loss_values = sent_probs_flat * (1 - bleu_scores_flat)  # [batch_size * (num_hypos + 1)]

        counters = dict(
            loss=tf.reduce_sum(loss_values),
            out_len=tf.to_float(tf.reduce_sum(batch['out_len'])),
        )
        append_counters_common_metrics(counters, logits, new_batch['out'], new_batch['out_len'], is_train)
        append_counters_xent(counters, loss_values, new_batch['out_len'])
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
