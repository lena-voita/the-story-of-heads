# Basic NN layers

import lib

import tensorflow as tf
from ..util import nop_ctx
from ..ops import record_activations as rec
from .lrp import LRP, relprop_add
from ..ops.basic import *

###############################################################################
#                                                                             #
#                                   LAYERS                                    #
#                                                                             #
###############################################################################



## ----------------------------------------------------------------------------
#                                   Dense
class Dense:
    def __init__(
            self, name,
            inp_size, out_size, activ=tf.tanh,
            matrix=None, bias=None,
            matrix_initializer=None, bias_initializer=None):

        """
        <name>/W
        <name>/b

        User can explicitly specify matrix to use instead of W (<name>/W is
        not created then), but this is not recommended to external users.
        """
        self.name = name
        self.activ = activ
        self.inp_size = inp_size
        self.out_size = out_size

        with tf.variable_scope(name) as self.scope:
            if matrix is None:
                self.W = get_model_variable('W', shape=[inp_size, out_size], initializer=matrix_initializer)
            else:
                self.W = matrix

            if bias is None:
                self.b = get_model_variable('b', shape=[out_size], initializer=bias_initializer)
            else:
                self.b = bias

    def __call__(self, inp):
        """
        inp: [..., inp_size]
        --------------------
        Ret: [..., out_size]
        """
        with tf.variable_scope(self.scope):
            out = self.activ(dot(inp, self.W) + self.b)
            out.set_shape([None] * (out.shape.ndims - 1) + [self.out_size])
            if rec.is_recorded():
                rec.save_activations(inp=inp, out=out)
            return out

    def relprop(self, output_relevance):
        """
        computes input relevance given output_relevance
        :param output_relevance: relevance w.r.t. layer output, [*dims, out_size]
        notation from DOI:10.1371/journal.pone.0130140, Eq 60
        """

        with tf.variable_scope(self.scope):
            inp, out = rec.get_activations('inp', 'out')
            # inp: [*dims, inp_size], out: [*dims, out_size]

            linear_self = Dense(self.name, self.inp_size, self.out_size, activ=nop, matrix=self.W, bias=self.b)

            # note: we apply relprop for each independent sample in order to avoid quadratic memory requirements
            flat_inp = tf.reshape(inp, [-1, tf.shape(inp)[-1]])
            flat_out_relevance = tf.reshape(output_relevance, [-1, tf.shape(output_relevance)[-1]])

            flat_inp_relevance = tf.map_fn(
                lambda i: LRP.relprop(linear_self, flat_out_relevance[i, None], flat_inp[i, None],
                                      jacobians=[tf.transpose(self.W)[None, :, None, :]], batch_axes=(0,))[0],
                elems=tf.range(tf.shape(flat_inp)[0]), dtype=inp.dtype, parallel_iterations=2 ** 10)
            input_relevance = LRP.rescale(output_relevance, tf.reshape(flat_inp_relevance, tf.shape(inp)))

        return input_relevance

    @property
    def input_size(self):
        return self.inp_size

    @property
    def output_size(self):
        return self.out_size

## ----------------------------------------------------------------------------
#                                 Embedding

class Embedding:
    def __init__(self, name, voc_size, emb_size, matrix=None, initializer=None, device=''):
        """
        Parameters:

          <name>/mat
        """
        self.name = name
        self.voc_size = voc_size
        self.emb_size = emb_size
        self.device = device

        if matrix is not None:
            self.mat = matrix
        else:
            with tf.variable_scope(name), (tf.device(device) if device is not None else nop_ctx()):
                self.mat = get_model_variable('mat', shape=[voc_size, emb_size], initializer=initializer)

    def __call__(self, inp, gumbel=False):
        """
        inp: [...]
        --------------------
        Ret: [..., emb_size]
        """
        with tf.name_scope(self.name), (tf.device(self.device) if self.device is not None else nop_ctx()):
            return tf.gather(self.mat, inp) if not gumbel else dot(inp, self.mat)

## ----------------------------------------------------------------------------
#                               LayerNorm

class LayerNorm:
    """
    Performs Layer Normalization
    """
    def __init__(self, name, inp_size, epsilon=1e-6):
        self.name = name
        self.epsilon = epsilon

        with tf.variable_scope(name):
            self.scale = get_model_variable('scale', shape=[inp_size], initializer=tf.ones_initializer())
            self.bias = get_model_variable('bias', shape=[inp_size], initializer=tf.zeros_initializer())

    def __call__(self, inp):
        with tf.variable_scope(self.name):
            mean = tf.reduce_mean(inp, axis=[-1], keep_dims=True)
            variance = tf.reduce_mean(tf.square(inp - mean), axis=[-1], keep_dims=True)
            norm_x = (inp - mean) * tf.rsqrt(variance + self.epsilon)
            out = norm_x * self.scale + self.bias
            if rec.is_recorded():
                rec.save_activations(inp=inp, out=out)
            return out

    def _jacobian(self, inp):
        assert inp.shape.ndims == 2, "Please reshape your inputs to [batch, dim]"
        batch_size = tf.shape(inp)[0]
        hid_size = tf.shape(inp)[1]
        centered_inp = (inp - tf.reduce_mean(inp, axis=[-1], keep_dims=True))
        variance = tf.reduce_mean(tf.square(centered_inp), axis=[-1], keep_dims=True)
        invstd_factor = tf.rsqrt(variance)

        # note: the code below will compute jacobian without taking self.scale into account until the _last_ line
        jac_out_wrt_invstd_factor = tf.reduce_sum(tf.diag(centered_inp), axis=-1, keepdims=True)
        jac_out_wrt_variance = jac_out_wrt_invstd_factor * (-0.5 * (variance + self.epsilon) ** (-1.5))[:, :, None,
                                                           None]
        jac_out_wrt_squared_difference = jac_out_wrt_variance * tf.fill([hid_size], 1. / tf.to_float(hid_size))

        hid_eye = tf.eye(hid_size, hid_size)[None, :, None, :]
        jac_out_wrt_centered_inp = tf.diag(
            invstd_factor) * hid_eye + jac_out_wrt_squared_difference * 2 * centered_inp
        jac_out_wrt_inp = jac_out_wrt_centered_inp - tf.reduce_mean(jac_out_wrt_centered_inp, axis=-1,
                                                                    keep_dims=True)
        return jac_out_wrt_inp * self.scale[None, :, None, None]

    def relprop(self, output_relevance):
        """
        computes input relevance given output_relevance
        :param output_relevance: relevance w.r.t. layer output, [*dims, out_size]
        notation from DOI:10.1371/journal.pone.0130140, Eq 60
        """
        with tf.variable_scope(self.name):
            inp, out = rec.get_activations('inp', 'out')
            # inp: [*dims, inp_size], out: [*dims, out_size]
            return self.relprop_explicit(output_relevance, inp)

    def relprop_explicit(self, output_relevance, inp):
        """ a version of relprop that accepts manually specified inputs instead of using collections """
        # note: we apply relprop for each independent sample in order to avoid quadratic memory requirements
        flat_inp = tf.reshape(inp, [-1, tf.shape(inp)[-1]])
        flat_out_relevance = tf.reshape(output_relevance, [-1, tf.shape(output_relevance)[-1]])

        flat_inp_relevance = tf.map_fn(
            lambda i: LRP.relprop(self, flat_out_relevance[i, None], flat_inp[i, None],
                                  jacobians=[self._jacobian(flat_inp[i, None])], batch_axes=(0,))[0],
            elems=tf.range(tf.shape(flat_inp)[0]), dtype=inp.dtype, parallel_iterations=2 ** 10)
        input_relevance = LRP.rescale(output_relevance, tf.reshape(flat_inp_relevance, tf.shape(inp)))
        return input_relevance

## ----------------------------------------------------------------------------
#                               ResidualWrapper


class Wrapper:
    """ Reflection-style wrapper, code from http://code.activestate.com/recipes/577555-object-wrapper-class/ """
    def __init__(self, wrapped_layer):
        self.wrapped_layer = wrapped_layer

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self.wrapped_layer, attr)


class ResidualLayerWrapper(Wrapper):
    def __init__(self, name, wrapped_layer, inp_size, out_size, steps='ldan', dropout=0, dropout_seed=None):
        """
        Applies any number of residual connection, dropout and/or layer normalization before or after wrapped layer
        :param steps: a sequence of operations to perform, containing any combination of:
            - 'l' - call wrapped [l]ayer, this operation should be used exactly once
            - 'd' - apply [d]ropout with p = dropout and seed = dropout_seed
            - 'a' - [a]dd inputs to output (residual connection)
            - 'n' - apply layer [n]ormalization here, can only be done once
        """
        assert steps.count('l') == 1, "residual wrapper must call wrapped layer exactly once"
        assert steps.count('n') <= 1, "in the current implementaion, there can be at most one layer normalization step"
        assert inp_size == out_size or 'a' not in steps, "residual step only works if inp_size == out_size"
        self.name = name
        self.wrapped_layer = wrapped_layer

        if 'n' in steps:
            ln_size = inp_size if steps.index('n') < steps.index('l') else out_size
            with tf.variable_scope(name) as self.scope:
                self.norm_layer = LayerNorm("layer_norm", ln_size)

        self.steps = steps
        self.preprocess_steps = steps[:steps.index('l')]
        self.postprocess_steps = steps[steps.index('l') + 1:]
        self.dropout = dropout
        self.dropout_seed = dropout_seed

    def __call__(self, inp, *args, **kwargs):
        out = self.preprocess(inp)
        out = self.wrapped_layer(out, *args, **kwargs)
        out = self.postprocess(out, inp)
        return out

    def preprocess(self, inp):
        return self._perform(self.preprocess_steps, inp)

    def postprocess(self, out, inp=None):
        return self._perform(self.postprocess_steps, out, inp=inp)

    def _perform(self, steps, out, inp=None):
        with tf.variable_scope(self.scope):
            if inp is None:
                inp = out

            for s in steps:
                if s == 'd':
                    if is_dropout_enabled():
                        out = lib.ops.dropout(out, 1.0 - self.dropout, seed=self.dropout_seed)
                elif s == 'a':
                    if rec.is_recorded():
                        rec.save_activations(residual_inp=inp, residual_update=out)
                    out = out + inp
                elif s == 'n':
                    out = self.norm_layer(out)
                else:
                    raise RuntimeError("Unknown process step: %s" % s)
            return out

    def relprop(self, R, main_key=None):
        with tf.variable_scope(self.scope):
            residual_inp, residual_update = rec.get_activations('residual_inp', 'residual_update')
        original_scale = tf.reduce_sum(abs(R))
        with tf.variable_scope(self.scope):
            Rinp_residual = 0.0
            for s in self.steps[::-1]:
                if s == 'l':
                    R = self.wrapped_layer.relprop(R)
                    if isinstance(R, dict):
                        assert main_key is not None
                        R_dict = R
                        R = R_dict[main_key]
                elif s == 'a':
                    # residual layer: LRP through addition
                    Rinp_residual, R = relprop_add(R, residual_inp, residual_update)
                elif s == 'n':
                    R = self.norm_layer.relprop(R)

            pre_residual_scale = tf.reduce_sum(abs(R) + abs(Rinp_residual))

            R = R + Rinp_residual
            R = R * pre_residual_scale / tf.reduce_sum(tf.abs(R))
            if main_key is not None:
                R_dict = dict(R_dict)
                R_dict[main_key] = R
                total_scale = sum(tf.reduce_sum(abs(relevance)) for relevance in R_dict.values())
                R_dict = {key: value * original_scale / total_scale
                          for key, value in R_dict.items()}
                return R_dict
            else:
                return R


###############################################################################
#                                                                             #
#                              SEQUENCE LOSSES                                #
#                                                                             #
###############################################################################


class SequenceLossBase:
    def rdo_to_logits(self, *args, **kwargs):
        raise NotImplementedError()

    def rdo_to_logits__predict(self, *args, **kwargs):
        return self.rdo_to_logits(*args, **kwargs)


class LossXent(SequenceLossBase):
    def __init__(
        self, name, rdo_size, voc, hp,
        matrix=None, bias=None,
        matrix_initializer=None, bias_initializer=tf.zeros_initializer(),
    ):
        """
        Parameters:

          Dense: <name>/logits
        """
        if 'lm_path' in hp:
            raise NotImplementedError("LM fusion not implemented")

        self.name = name
        self.rdo_size = rdo_size
        self.voc_size = voc.size()

        self.bos = voc.bos
        self.label_smoothing = hp.get('label_smoothing', 0)

        with tf.variable_scope(name):
            self._rdo_to_logits = Dense(
                'logits', rdo_size, self.voc_size, activ=nop,
                matrix=matrix, bias=bias,
                matrix_initializer=matrix_initializer, bias_initializer=bias_initializer)

    def __call__(self, rdo, out, out_len):
        """
        rdo: [batch_size, ninp, rdo_size]
        out: [batch_size, ninp], dtype=int
        out_len: [batch_size]
        inp_words: [batch_size, ninp], dtype=string
        attn_P_argmax: [batch_size, ninp], dtype=int
        --------------------------
        Ret: [batch_size]
        """
        logits = self.rdo_to_logits(rdo, out, out_len) # [batch_size, ninp, voc_size]
        return self.logits2loss(logits, out, out_len)

    def rdo_to_logits(self, rdo, out, out_len):
        """
        compute logits in training mode
        :param rdo: pre-final activations float32[batch, num_outputs, hid_size]
        :param out: output sequence, padded with EOS int64[batch, num_outputs]
        :param out_len: lengths of outputs in :out: excluding padding, int64[batch]
        """
        return self._rdo_to_logits(rdo)

    def logits2loss(self, logits, out, out_len, reduce_rows=True):
        if self.label_smoothing:
            voc_size = tf.shape(logits)[-1]
            smooth_positives = 1.0 - self.label_smoothing
            smooth_negatives = self.label_smoothing / tf.to_float(voc_size - 1)
            onehot_labels = tf.one_hot(out, depth=voc_size, on_value=smooth_positives, off_value=smooth_negatives)

            losses = tf.nn.softmax_cross_entropy_with_logits(
                labels=onehot_labels,
                logits=logits,
                name="xentropy")

            # Normalizing constant is the best cross-entropy value with soft targets.
            # We subtract it just for readability, makes no difference on learning.
            normalizing = -(smooth_positives * tf.log(smooth_positives) +
                tf.to_float(voc_size - 1) * smooth_negatives * tf.log(smooth_negatives + 1e-20))
            losses -= normalizing
        else:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=out)

        losses *= tf.sequence_mask(out_len, maxlen=tf.shape(out)[1], dtype=logits.dtype)

        if reduce_rows:
            return tf.reduce_sum(losses, axis=1)
        else:
            return losses

    def rdo_to_logits__predict(self, rdo, prefix):
        """ like rdo_to_logits, but used in beam search """
        return self._rdo_to_logits(rdo)


LossXentLm = LossXent  # alias


class FFN:
    """
    Feed-forward layer
    """

    def __init__(self, name,
                 inp_size, hid_size, out_size,
                 relu_dropout):
        assert isinstance(hid_size, int), "List of hidden sizes not is not supported"
        self.name = name
        self.relu_dropout = relu_dropout

        with tf.variable_scope(name):
            self.first_conv = Dense(
                'conv1',
                inp_size, hid_size,
                activ=tf.nn.relu,
                bias_initializer=tf.zeros_initializer())

            self.second_conv = Dense(
                'conv2',
                hid_size, out_size,
                activ=lambda x: x,
                bias_initializer=tf.zeros_initializer())

    def __call__(self, inputs, summarize_preactivations=False):
        """
        inp: [batch_size * ninp * inp_dim]
        ---------------------------------
        out: [batch_size * ninp * out_dim]
        """
        with tf.variable_scope(self.name):
            hidden = self.first_conv(inputs)
            if is_dropout_enabled():
                hidden = dropout(hidden, 1.0 - self.relu_dropout)

            outputs = self.second_conv(hidden)

        return outputs

    def relprop(self, R):
        R = self.second_conv.relprop(R)
        R = self.first_conv.relprop(R)
        return R