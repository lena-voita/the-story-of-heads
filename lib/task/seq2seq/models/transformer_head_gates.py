#!/usr/bin/env python3
from lib.layers import *
from lib.ops import *
from ..models import TranslateModelBase, TranslateModel
from ..data import *
from collections import namedtuple


class Transformer:
    def __init__(
            self, name,
            inp_voc, out_voc,
            *_args,
            emb_size=None, hid_size=512,
            key_size=None, value_size=None,
            inner_hid_size=None,  # DEPRECATED. Left for compatibility with older experiments
            ff_size=None,
            num_heads=8, num_layers=6,
            attn_dropout=0.0, attn_value_dropout=0.0, relu_dropout=0.0, res_dropout=0.1,
            share_emb=False, inp_emb_bias=False, rescale_emb=False,
            dst_reverse=False, dst_rand_offset=False, summarize_preactivations=False,
            res_steps='nlda', normalize_out=False, multihead_attn_format='v1',
            emb_inp_device='', emb_out_device='',
            concrete_heads={},  # any subset of {enc-self, dec-self, dec-enc}
            alive_heads={},  # {enc-self: [[1,1,1,0,1,0,0,0], [0,0,0,1,0,1,0,1], ..., [0,1,0,1,0,0,0,0]],
                             # dec-self: [...],
                             # dec-enc: [...]}
            num_layers_enc=0,
            num_layers_dec=0,
            **_kwargs
    ):

        for attn_type in ['enc-self', 'dec-self', 'dec-enc']:
            assert not (attn_type in concrete_heads and attn_type in alive_heads),\
                "'{}' is passed as both with trainable concrete gates heads and fixed gates".format(attn_type)

        if isinstance(ff_size, str):
            ff_size = [int(i) for i in ff_size.split(':')]

        if _args:
            raise Exception("Unexpected positional arguments")

        emb_size = emb_size if emb_size else hid_size
        key_size = key_size if key_size else hid_size
        value_size = value_size if value_size else hid_size
        if key_size % num_heads != 0:
            raise Exception("Bad number of heads")
        if value_size % num_heads != 0:
            raise Exception("Bad number of heads")

        self.name = name
        self.num_layers_enc = num_layers if num_layers_enc == 0 else num_layers_enc
        self.num_layers_dec = num_layers if num_layers_dec == 0 else num_layers_dec
        self.res_dropout = res_dropout
        self.emb_size = emb_size
        self.hid_size = hid_size
        self.rescale_emb = rescale_emb
        self.summarize_preactivations = summarize_preactivations
        self.dst_reverse = dst_reverse
        self.dst_rand_offset = dst_rand_offset
        self.normalize_out = normalize_out

        with tf.variable_scope(name):
            max_voc_size = max(inp_voc.size(), out_voc.size())

            self.emb_inp = Embedding(
                'emb_inp', max_voc_size if share_emb else inp_voc.size(), emb_size,
                initializer=tf.random_normal_initializer(0, emb_size ** -.5),
                device=emb_inp_device)

            self.emb_out = Embedding(
                'emb_out', max_voc_size if share_emb else out_voc.size(), emb_size,
                matrix=self.emb_inp.mat if share_emb else None,
                initializer=tf.random_normal_initializer(0, emb_size ** -.5),
                device=emb_out_device)

            self.emb_inp_bias = 0
            if inp_emb_bias:
                self.emb_inp_bias = get_model_variable('emb_inp_bias', shape=[1, 1, emb_size])

            def get_layer_params(layer_prefix, layer_idx):
                layer_name = '%s-%i' % (layer_prefix, layer_idx)
                inp_out_size = emb_size if layer_idx == 0 else hid_size
                return layer_name, inp_out_size

            def attn_layer(layer_prefix, layer_idx, **kwargs):
                layer_name, inp_out_size = get_layer_params(layer_prefix, layer_idx)
                return ResidualLayerWrapper(
                    layer_name,
                    MultiHeadAttn(
                        layer_name,
                        inp_size=inp_out_size,
                        key_depth=key_size,
                        value_depth=value_size,
                        output_depth=hid_size,
                        num_heads=num_heads,
                        attn_dropout=attn_dropout,
                        attn_value_dropout=attn_value_dropout,
                        **kwargs),
                    inp_size=inp_out_size,
                    out_size=inp_out_size,
                    steps=res_steps,
                    dropout=res_dropout)

            def attn_layer_concrete_heads(layer_prefix, layer_idx, **kwargs):
                layer_name, inp_out_size = get_layer_params(layer_prefix, layer_idx)
                return ResidualLayerWrapper(
                    layer_name,
                    MultiHeadAttnConcrete(
                        layer_name,
                        inp_size=inp_out_size,
                        key_depth=key_size,
                        value_depth=value_size,
                        output_depth=hid_size,
                        num_heads=num_heads,
                        attn_dropout=attn_dropout,
                        attn_value_dropout=attn_value_dropout,
                        **kwargs),
                    inp_size=inp_out_size,
                    out_size=inp_out_size,
                    steps=res_steps,
                    dropout=res_dropout)

            def attn_layer_fixed_alive_heads(layer_prefix, layer_idx, head_gate, **kwargs):
                layer_name, inp_out_size = get_layer_params(layer_prefix, layer_idx)
                return ResidualLayerWrapper(
                    layer_name,
                    MultiHeadAttnFixedAliveHeads(
                        layer_name,
                        inp_size=inp_out_size,
                        key_depth=key_size,
                        value_depth=value_size,
                        output_depth=hid_size,
                        num_heads=num_heads,
                        attn_dropout=attn_dropout,
                        attn_value_dropout=attn_value_dropout,
                        head_gate=head_gate,
                        **kwargs),
                    inp_size=inp_out_size,
                    out_size=inp_out_size,
                    steps=res_steps,
                    dropout=res_dropout)

            def ffn_layer(layer_prefix, layer_idx, ffn_hid_size):
                layer_name, inp_out_size = get_layer_params(layer_prefix, layer_idx)
                return ResidualLayerWrapper(
                    layer_name,
                    FFN(
                        layer_name,
                        inp_size=inp_out_size,
                        hid_size=ffn_hid_size,
                        out_size=hid_size,
                        relu_dropout=relu_dropout),
                    inp_size=inp_out_size,
                    out_size=hid_size,
                    steps=res_steps,
                    dropout=res_dropout)

            # Encoder/decoder layer params
            enc_ffn_hid_size = ff_size if ff_size else (inner_hid_size if inner_hid_size else hid_size)
            dec_ffn_hid_size = ff_size if ff_size else hid_size
            dec_enc_attn_format = 'use_kv' if multihead_attn_format == 'v1' else 'combined'

            # Encoder Layers
            self.enc_attn = [attn_layer_concrete_heads('enc_attn', i) if 'enc-self' in concrete_heads else
                             attn_layer('enc_attn', i) if not 'enc-self' in alive_heads else
                             attn_layer_fixed_alive_heads('enc_attn', i, alive_heads['enc-self'][i])
                             for i in range(self.num_layers_enc)]

            self.enc_ffn = [ffn_layer('enc_ffn', i, enc_ffn_hid_size) for i in range(self.num_layers_enc)]

            if self.normalize_out:
                self.enc_out_norm = LayerNorm('enc_out_norm',
                                              inp_size=emb_size if self.num_layers_enc == 0 else hid_size)

            # Decoder layers
            self.dec_attn = [attn_layer_concrete_heads('dec_attn', i) if 'dec-self' in concrete_heads else
                             attn_layer('dec_attn', i) if not 'dec-self' in alive_heads else
                             attn_layer_fixed_alive_heads('dec_attn', i, alive_heads['dec-self'][i])
                             for i in range(self.num_layers_dec)]

            self.dec_enc_attn = [attn_layer_concrete_heads('dec_enc_attn', i, _format=dec_enc_attn_format) \
                                 if 'dec-enc' in concrete_heads else \
                             attn_layer('dec_enc_attn', i, _format=dec_enc_attn_format) if \
                    not 'dec-enc' in alive_heads else \
                             attn_layer_fixed_alive_heads('dec_enc_attn', i, alive_heads['dec-enc'][i], _format=dec_enc_attn_format)
                             for i in range(self.num_layers_enc)]

            self.dec_ffn = [ffn_layer('dec_ffn', i, dec_ffn_hid_size) for i in range(self.num_layers_dec)]

            if self.normalize_out:
                self.dec_out_norm = LayerNorm('dec_out_norm',
                                              inp_size=emb_size if self.num_layers_dec == 0 else hid_size)

    def encode(self, inp, inp_len, is_train):
        with dropout_scope(is_train), tf.name_scope('mod_enc') as scope:

            # Embeddings
            emb_inp = self.emb_inp(inp)  # [batch_size * ninp * emb_dim]
            if self.rescale_emb:
                emb_inp *= self.emb_size ** .5
            emb_inp += self.emb_inp_bias

            # Prepare decoder
            enc_attn_mask = self._make_enc_attn_mask(inp, inp_len)  # [batch_size * 1 * 1 * ninp]

            enc_inp = self._add_timing_signal(emb_inp)

            # Apply dropouts
            if is_dropout_enabled():
                enc_inp = tf.nn.dropout(enc_inp, 1.0 - self.res_dropout)

            # Encoder
            for layer in range(self.num_layers_enc):
                enc_inp = self.enc_attn[layer](enc_inp, enc_attn_mask)
                enc_inp = self.enc_ffn[layer](enc_inp, summarize_preactivations=self.summarize_preactivations)

            if self.normalize_out:
                enc_inp = self.enc_out_norm(enc_inp)

            tf.add_to_collection(lib.meta.ACTIVATIONS, tf.identity(enc_inp, name=scope))

            return enc_inp, enc_attn_mask

    def decode(self, out, out_len, out_reverse, enc_out, enc_attn_mask, is_train):
        with dropout_scope(is_train), tf.name_scope('mod_dec') as scope:
            # Embeddings
            emb_out = self.emb_out(out)  # [batch_size * nout * emb_dim]
            if self.rescale_emb:
                emb_out *= self.emb_size ** .5

            # Shift right; drop embedding for last word
            emb_out = tf.pad(emb_out, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]

            # Prepare decoder
            dec_attn_mask = self._make_dec_attn_mask(out)  # [1 * 1 * nout * nout]

            offset = 'random' if self.dst_rand_offset else 0
            dec_inp = self._add_timing_signal(emb_out, offset=offset, inp_reverse=out_reverse)
            # Apply dropouts
            if is_dropout_enabled():
                dec_inp = tf.nn.dropout(dec_inp, 1.0 - self.res_dropout)

            # bypass info from Encoder to avoid None gradients for num_layers_dec == 0
            if self.num_layers_dec == 0:
                inp_mask = tf.squeeze(tf.transpose(enc_attn_mask, perm=[3, 1, 2, 0]), 3)
                dec_inp += tf.reduce_mean(enc_out * inp_mask, axis=[0, 1], keep_dims=True)

            # Decoder
            for layer in range(self.num_layers_dec):
                dec_inp = self.dec_attn[layer](dec_inp, dec_attn_mask)
                dec_inp = self.dec_enc_attn[layer](dec_inp, enc_attn_mask, enc_out)
                dec_inp = self.dec_ffn[layer](dec_inp, summarize_preactivations=self.summarize_preactivations)

            if self.normalize_out:
                dec_inp = self.dec_out_norm(dec_inp)

            tf.add_to_collection(lib.meta.ACTIVATIONS, tf.identity(dec_inp, name=scope))

            return dec_inp

    def _make_enc_attn_mask(self, inp, inp_len, dtype=tf.float32):
        """
        inp = [batch_size * ninp]
        inp_len = [batch_size]

        attn_mask = [batch_size * 1 * 1 * ninp]
        """
        with tf.variable_scope("make_enc_attn_mask"):
            inp_mask = tf.sequence_mask(inp_len, dtype=dtype, maxlen=tf.shape(inp)[1])

            attn_mask = inp_mask[:, None, None, :]
            return attn_mask

    def _make_dec_attn_mask(self, out, dtype=tf.float32):
        """
        out = [baatch_size * nout]

        attn_mask = [1 * 1 * nout * nout]
        """
        with tf.variable_scope("make_dec_attn_mask"):
            length = tf.shape(out)[1]
            lower_triangle = tf.matrix_band_part(tf.ones([length, length], dtype=dtype), -1, 0)
            attn_mask = tf.reshape(lower_triangle, [1, 1, length, length])
            return attn_mask

    def _add_timing_signal(self, inp, min_timescale=1.0, max_timescale=1.0e4, offset=0, inp_reverse=None):
        """
        inp: (batch_size * ninp * hid_dim)
        :param offset: add this number to all character positions.
            if offset == 'random', picks this number uniformly from [-32000,32000] integers
        :type offset: number, tf.Tensor or 'random'
        """
        with tf.variable_scope("add_timing_signal"):
            ninp = tf.shape(inp)[1]
            hid_size = tf.shape(inp)[2]

            position = tf.to_float(tf.range(ninp))[None, :, None]

            if offset == 'random':
                BIG_LEN = 32000
                offset = tf.random_uniform(tf.shape(position), minval=-BIG_LEN, maxval=BIG_LEN, dtype=tf.int32)

            # force broadcasting over batch axis
            if isinstance(offset * 1, tf.Tensor):  # multiply by 1 to also select variables, special generators, etc.
                assert offset.shape.ndims in (0, 1, 2)
                new_shape = [tf.shape(offset)[i] for i in range(offset.shape.ndims)]
                new_shape += [1] * (3 - len(new_shape))
                offset = tf.reshape(offset, new_shape)

            position += tf.to_float(offset)

            if inp_reverse is not None:
                position = tf.multiply(
                    position,
                    tf.where(
                        tf.equal(inp_reverse, 0),
                        tf.ones_like(inp_reverse, dtype=tf.float32),
                        -1.0 * tf.ones_like(inp_reverse, dtype=tf.float32)
                    )[:, None, None]  # (batch_size * ninp * dim)
                )
            num_timescales = hid_size // 2
            log_timescale_increment = (
                    math.log(float(max_timescale) / float(min_timescale)) /
                    (tf.to_float(num_timescales) - 1))
            inv_timescales = min_timescale * tf.exp(
                tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)

            # scaled_time: [ninp * hid_dim]
            scaled_time = position * inv_timescales[None, None, :]
            signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=-1)
            signal = tf.pad(signal, [[0, 0], [0, 0], [0, tf.mod(hid_size, 2)]])
            return inp + signal


# ============================================================================
#                                  Transformer model

class Model(TranslateModelBase):

    def __init__(self, name, inp_voc, out_voc, **hp):
        self.name = name
        self.inp_voc = inp_voc
        self.out_voc = out_voc
        self.hp = hp

        # Parameters
        self.transformer = Transformer(name, inp_voc, out_voc, **hp)

        projection_matrix = None
        if hp.get('dwwt', False):
            projection_matrix = tf.transpose(self.transformer.emb_out.mat)

        self.loss = LossXent(
            hp.get('loss_name', 'loss_xent_lm'),
            hp['hid_size'],
            out_voc,
            hp,
            matrix=projection_matrix,
            bias=None if hp.get("loss_bias", False) else 0)

        inference_mode = hp.get("inference_mode", "fast")
        if inference_mode == 'fast':
            self.translate_model = TranslateModelFast(self.name, self.transformer, self.loss, self.inp_voc,
                                                      self.out_voc)
        elif inference_mode == 'lazy':
            self.translate_model = TranslateModelLazy(self.name, self.transformer, self.loss, self.inp_voc,
                                                      self.out_voc)
        else:
            raise NotImplementedError("inference_mode %s is not supported" % inference_mode)

    # Train interface
    def encode_decode(self, batch, is_train, score_info=False):
        inp = batch['inp']  # [batch_size * ninp]
        out = batch['out']  # [batch_size * nout]
        inp_len = batch.get('inp_len', infer_length(inp, self.inp_voc.eos, time_major=False))  # [batch]
        out_len = batch.get('out_len', infer_length(out, self.out_voc.eos, time_major=False))  # [batch]

        out_reverse = tf.zeros_like(inp_len)  # batch['out_reverse']

        # rdo: [batch_size * nout * hid_dim]
        enc_out, enc_attn_mask = self.transformer.encode(inp, inp_len, is_train)
        rdo = self.transformer.decode(out, out_len, out_reverse, enc_out, enc_attn_mask, is_train)

        return rdo

    def make_feed_dict(self, batch, **kwargs):
        feed_dict = make_batch_data(batch, self.inp_voc, self.out_voc,
                                    force_bos=self.hp.get('force_bos', False),
                                    **kwargs)
        return feed_dict



    # ======== TranslateModel for Inference ============
    def encode(self, batch, **flags):
        """
        :param batch: a dict of {string:symbolic tensor} that model understands.
            By default it should accept {'inp': int32 matrix[batch,time]}
        :return: initial decoder state
        """
        return self.translate_model.encode(batch, **flags)

    def decode(self, dec_state, words=None, **flags):
        """
        Performs decoding step given words and previous state.
        :param words: previous output tokens, int32[batch_size]. if None, uses zero embeddings (first step)
        :returns: next state
        """
        return self.translate_model.decode(dec_state, words, **flags)

    def sample(self, dec_state, base_scores, slices, k, **kwargs):
        return self.translate_model.sample(dec_state, base_scores, slices, k, **kwargs)

    def get_rdo(self, dec_state, **kwargs):
        return self.translate_model.get_rdo(dec_state, **kwargs)

    def get_attnP(self, dec_state, **kwargs):
        return self.translate_model.get_attnP(dec_state, **kwargs)


class ScopedModel(Model):

    def __init__(self, name, inp_voc, out_voc, **hp):
        with tf.variable_scope(name):
            super(ScopedModel, self).__init__(name, inp_voc, out_voc, **hp)

    def encode_decode(self, *args, **kwargs):
        with tf.name_scope(self.name):
            return super(ScopedModel, self).encode_decode(*args, **kwargs)

    def encode(self, *args, **kwargs):
        with tf.name_scope(self.name):
            return super(ScopedModel, self).encode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        with tf.name_scope(self.name):
            return super(ScopedModel, self).decode(*args, **kwargs)

    def sample(self, *args, **kwargs):
        with tf.name_scope(self.name):
            return super(ScopedModel, self).sample(*args, **kwargs)


# ============================================================================
#                              Transformer inference

class TranslateModelFast(TranslateModel):
    DecState = namedtuple("transformer_state", ['enc_out', 'enc_attn_mask', 'attnP', 'rdo', 'out_seq', 'offset',
                                                'emb', 'dec_layers', 'dec_enc_kv', 'dec_dec_kv'])

    def __init__(self, name, transformer, loss, inp_voc, out_voc):
        """
        A translation model that performs quick (n^2) inference for transformer
        with manual implementation of 1-step decoding
        """
        self.name = name
        self.transformer = transformer
        self.loss = loss
        self.inp_voc = inp_voc
        self.out_voc = out_voc

    def encode(self, batch, is_train=False, **kwargs):
        """
        :param batch: a dict containing 'inp':int32[batch_size * ninp] and optionally inp_len:int32[batch_size]
        :param is_train: if True, enables dropouts
        """
        inp = batch['inp']
        inp_len = batch.get('inp_len', infer_length(inp, self.inp_voc.eos, time_major=False))
        with dropout_scope(is_train), tf.name_scope(self.transformer.name):
            # Encode.
            enc_out, enc_attn_mask = self.transformer.encode(inp, inp_len, is_train=False)

            # Decoder dummy input/output
            ninp = tf.shape(inp)[1]
            batch_size = tf.shape(inp)[0]
            hid_size = tf.shape(enc_out)[-1]
            out_seq = tf.zeros([batch_size, 0], dtype=inp.dtype)
            rdo = tf.zeros([batch_size, hid_size], dtype=enc_out.dtype)

            attnP = tf.ones([batch_size, ninp]) / tf.to_float(inp_len)[:, None]

            offset = tf.zeros((batch_size,))
            if self.transformer.dst_rand_offset:
                BIG_LEN = 32000
                random_offset = tf.random_uniform(tf.shape(offset), minval=-BIG_LEN, maxval=BIG_LEN, dtype=tf.int32)
                offset += tf.to_float(random_offset)

            trans = self.transformer
            empty_emb = tf.zeros([batch_size, 0, trans.emb_size])
            empty_dec_layers = [tf.zeros([batch_size, 0, trans.hid_size])] * trans.num_layers_dec
            input_layers = [empty_emb] + empty_dec_layers[:-1]

            # prepare kv parts for all decoder attention layers. Note: we do not preprocess enc_out
            # for each layer because ResidualLayerWrapper only preprocesses first input (query)
            dec_enc_kv = [layer.kv_conv(enc_out)
                          for i, layer in enumerate(trans.dec_enc_attn)]
            dec_dec_kv = [layer.kv_conv(layer.preprocess(input_layers[i]))
                          for i, layer in enumerate(trans.dec_attn)]

            new_state = self.DecState(enc_out, enc_attn_mask, attnP, rdo, out_seq, offset,
                                      empty_emb, empty_dec_layers, dec_enc_kv, dec_dec_kv)

            # perform initial decode (instead of force_bos) with zero embeddings
            new_state = self.decode(new_state, is_train=is_train)
            return new_state

    def decode(self, dec_state, words=None, is_train=False, **kwargs):
        """
        Performs decoding step given words and previous state.
        Returns next state.

        :param words: previous output tokens, int32[batch_size]. if None, uses zero embeddings (first step)
        :param is_train: if True, enables dropouts
        """
        trans = self.transformer
        enc_out, enc_attn_mask, attnP, rdo, out_seq, offset, prev_emb = dec_state[:7]
        prev_dec_layers = dec_state.dec_layers
        dec_enc_kv = dec_state.dec_enc_kv
        dec_dec_kv = dec_state.dec_dec_kv

        batch_size = tf.shape(rdo)[0]
        if words is not None:
            out_seq = tf.concat([out_seq, tf.expand_dims(words, 1)], 1)

        with dropout_scope(is_train), tf.name_scope(trans.name):
            # Embeddings
            if words is None:
                # initial step: words are None
                emb_out = tf.zeros((batch_size, 1, trans.emb_size))
            else:
                emb_out = trans.emb_out(words[:, None])  # [batch_size * 1 * emb_dim]
                if trans.rescale_emb:
                    emb_out *= trans.emb_size ** .5

            # Prepare decoder
            dec_inp_t = trans._add_timing_signal(emb_out, offset=offset)
            # Apply dropouts
            if is_dropout_enabled():
                dec_inp_t = tf.nn.dropout(dec_inp_t, 1.0 - trans.res_dropout)

            # bypass info from Encoder to avoid None gradients for num_layers_dec == 0
            if trans.num_layers_dec == 0:
                inp_mask = tf.squeeze(tf.transpose(enc_attn_mask, perm=[3, 1, 2, 0]), 3)
                dec_inp_t += tf.reduce_mean(enc_out * inp_mask, axis=[0, 1], keep_dims=True)

            # Decoder
            new_emb = tf.concat([prev_emb, dec_inp_t], axis=1)
            _out = tf.pad(out_seq, [(0, 0), (0, 1)])
            dec_attn_mask = trans._make_dec_attn_mask(_out)[:, :, -1:, :]  # [1, 1, n_q=1, n_kv]

            new_dec_layers = []
            new_dec_dec_kv = []

            for layer in range(trans.num_layers_dec):
                # multi-head self-attention: use only the newest time-step as query,
                # but all time-steps up to newest one as keys/values
                next_dec_kv = trans.dec_attn[layer].kv_conv(trans.dec_attn[layer].preprocess(dec_inp_t))
                new_dec_dec_kv.append(tf.concat([dec_dec_kv[layer], next_dec_kv], axis=1))
                dec_inp_t = trans.dec_attn[layer](dec_inp_t, dec_attn_mask, kv=new_dec_dec_kv[layer])

                dec_inp_t = trans.dec_enc_attn[layer](dec_inp_t, enc_attn_mask, kv=dec_enc_kv[layer])
                dec_inp_t = trans.dec_ffn[layer](dec_inp_t, summarize_preactivations=trans.summarize_preactivations)

                new_dec_inp = tf.concat([prev_dec_layers[layer], dec_inp_t], axis=1)
                new_dec_layers.append(new_dec_inp)

            if trans.normalize_out:
                dec_inp_t = trans.dec_out_norm(dec_inp_t)

            rdo = dec_inp_t[:, -1]

            new_state = self.DecState(enc_out, enc_attn_mask, attnP, rdo, out_seq, offset + 1,
                                      new_emb, new_dec_layers, dec_enc_kv, new_dec_dec_kv)
            return new_state

    def get_rdo(self, dec_state, **kwargs):
        return dec_state.rdo, dec_state.out_seq

    def get_attnP(self, dec_state, **kwargs):
        return dec_state.attnP


class TranslateModelLazy(TranslateModel):
    def __init__(self, name, transformer, loss, inp_voc, out_voc):
        """
        Automatically implements O(n^3) decoding by using trans.decode
        """
        self.name = name
        self.transformer = transformer
        self.loss = loss
        self.inp_voc = inp_voc
        self.out_voc = out_voc

    def encode(self, batch, is_train=False, **kwargs):
        """
            :param batch: a dict of placeholders
                inp: [batch_size * ninp]
                inp_len; [batch_size]
        """
        inp = batch['inp']
        inp_len = batch['inp_len']
        with dropout_scope(is_train), tf.name_scope(self.transformer.name):
            # Encode.
            enc_out, enc_attn_mask = self.transformer.encode(inp, inp_len, is_train=False)

            # Decoder dummy input/output
            ninp = tf.shape(inp)[1]
            batch_size = tf.shape(inp)[0]
            hid_size = tf.shape(enc_out)[-1]
            out_seq = tf.zeros([batch_size, 0], dtype=inp.dtype)
            rdo = tf.zeros([batch_size, hid_size], dtype=enc_out.dtype)

            attnP = tf.ones([batch_size, ninp]) / tf.to_float(inp_len)[:, None]

            return self._decode_impl((enc_out, enc_attn_mask, attnP, out_seq, rdo), **kwargs)

    def decode(self, dec_state, words, **kwargs):
        """
        Performs decoding step given words

        words: [batch_size]
        """
        with tf.name_scope(self.transformer.name):
            (enc_out, enc_attn_mask, attnP, prev_out_seq, rdo) = dec_state
            out_seq = tf.concat([prev_out_seq, tf.expand_dims(words, 1)], 1)
            return self._decode_impl((enc_out, enc_attn_mask, attnP, out_seq, rdo), **kwargs)

    def _decode_impl(self, dec_state, is_train=False, **kwargs):
        (enc_out, enc_attn_mask, attnP, out_seq, rdo) = dec_state

        with dropout_scope(is_train):
            out = tf.pad(out_seq, [(0, 0), (0, 1)])
            out_len = tf.fill(dims=(tf.shape(out)[0],), value=tf.shape(out_seq)[1])
            out_reverse = tf.zeros_like(out_len)  # batch['out_reverse']
            dec_out = self.transformer.decode(out, out_len, out_reverse, enc_out, enc_attn_mask, is_train=False)
            rdo = dec_out[:, -1, :]  # [batch_size * hid_dim]

            attnP = enc_attn_mask[:, 0, 0, :]  # [batch_size * ninp ]
            attnP /= tf.reduce_sum(attnP, axis=1, keep_dims=True)

            return (enc_out, enc_attn_mask, attnP, out_seq, rdo)

    def get_rdo(self, dec_state, **kwargs):
        rdo = dec_state[4]
        out = dec_state[3]
        return rdo, out

    def get_attnP(self, dec_state, **kwargs):
        return dec_state[2]

