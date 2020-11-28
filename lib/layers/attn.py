
import tensorflow as tf
import math

import lib
from lib.ops.basic import is_dropout_enabled, dropout
from lib.ops import record_activations as rec
from .basic import Dense

from lib.layers.concrete_gate import ConcreteGate


class MultiHeadAttn:
    """
    Multihead scaled-dot-product attention with input/output transformations
    """
    ATTN_BIAS_VALUE = -1e9

    def __init__(
            self, name, inp_size,
            key_depth, value_depth, output_depth,
            num_heads, attn_dropout, attn_value_dropout,
            kv_inp_size=None, _format='combined'
    ):
        self.name = name
        self.key_depth = key_depth
        self.value_depth = value_depth
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.attn_value_dropout = attn_value_dropout
        self.format = _format
        kv_inp_size = kv_inp_size or inp_size

        with tf.variable_scope(name) as scope:
            self.scope = scope

            if self.format == 'use_kv':
                self.query_conv = Dense(
                    'query_conv',
                    inp_size, key_depth,
                    activ=lambda x: x,
                    bias_initializer=tf.zeros_initializer(),
                )

                self.kv_conv = Dense(
                    'mem_conv',
                    kv_inp_size, key_depth + value_depth,
                    activ=lambda x: x,
                    bias_initializer=tf.zeros_initializer(),
                )

                if kv_inp_size == inp_size:
                    self.combined_conv = Dense(
                        'combined_conv',
                        inp_size, key_depth * 2 + value_depth,
                        activ=lambda x: x,
                        matrix=tf.concat([self.query_conv.W, self.kv_conv.W], axis=1),
                        bias=tf.concat([self.query_conv.b, self.kv_conv.b], axis=0),
                    )

            elif self.format == 'combined':
                assert inp_size == kv_inp_size, 'combined format is only supported when inp_size == kv_inp_size'
                self.combined_conv = Dense(
                    'mem_conv',  # old name for compatibility
                    inp_size, key_depth * 2 + value_depth,
                    activ=lambda x: x,
                    bias_initializer=tf.zeros_initializer())

                self.query_conv = Dense(
                    'query_conv',
                    inp_size, key_depth,
                    activ=lambda x: x,
                    matrix=self.combined_conv.W[:, :key_depth],
                    bias=self.combined_conv.b[:key_depth],
                )

                self.kv_conv = Dense(
                    'kv_conv',
                    kv_inp_size, key_depth + value_depth,
                    activ=lambda x: x,
                    matrix=self.combined_conv.W[:, key_depth:],
                    bias=self.combined_conv.b[key_depth:],
                )
            else:
                raise Exception("Unexpected format: " + self.format)

            self.out_conv = Dense(
                'out_conv',
                value_depth, output_depth,
                activ=lambda x: x,
                bias_initializer=tf.zeros_initializer())

    def attention_core(self, q, k, v, attn_mask):
        """ Core math operations of multihead attention layer """
        q = self._split_heads(q)  # [batch_size * n_heads * n_q * (k_dim/n_heads)]
        k = self._split_heads(k)  # [batch_size * n_heads * n_kv * (k_dim/n_heads)]
        v = self._split_heads(v)  # [batch_size * n_heads * n_kv * (v_dim/n_heads)]

        key_depth_per_head = self.key_depth / self.num_heads
        q = q / math.sqrt(key_depth_per_head)

        # Dot-product attention
        # logits: (batch_size * n_heads * n_q * n_kv)
        attn_bias = MultiHeadAttn.ATTN_BIAS_VALUE * (1 - attn_mask)
        logits = tf.matmul(
            q,
            tf.transpose(k, perm=[0, 1, 3, 2])) + attn_bias
        weights = tf.nn.softmax(logits)

        tf.add_to_collection("AttnWeights", weights)
        tf.add_to_collection(lib.meta.ATTENTIONS, lib.meta.Attention(self.scope, weights, logits, attn_mask))

        if is_dropout_enabled():
            weights = dropout(weights, 1.0 - self.attn_dropout)
        x = tf.matmul(
            weights,  # [batch_size * n_heads * n_q * n_kv]
            v  # [batch_size * n_heads * n_kv * (v_deph/n_heads)]
        )
        combined_x = self._combine_heads(x)

        if is_dropout_enabled():
            combined_x = dropout(combined_x, 1.0 - self.attn_value_dropout)
        return combined_x

    def __call__(self, query_inp, attn_mask, kv_inp=None, kv=None):
        """
        query_inp: [batch_size * n_q * inp_dim]
        attn_mask: [batch_size * 1 * n_q * n_kv]
        kv_inp: [batch_size * n_kv * inp_dim]
        -----------------------------------------------
        results: [batch_size * n_q * output_depth]
        """
        assert kv is None or kv_inp is None, "please only feed one of kv or kv_inp"

        with tf.variable_scope(self.scope), tf.name_scope(self.name) as scope:
            rec.save_activation('kv', kv)
            if kv_inp is not None or kv is not None:
                q = self.query_conv(query_inp)
                if kv is None:
                    kv = self.kv_conv(kv_inp)
                k, v = tf.split(kv, [self.key_depth, self.value_depth], axis=2)
                rec.save_activation('is_combined', False)
            else:
                combined = self.combined_conv(query_inp)
                q, k, v = tf.split(combined, [self.key_depth, self.key_depth, self.value_depth], axis=2)
                rec.save_activation('is_combined', True)

            rec.save_activations(q=q, k=k, v=v, attn_mask=attn_mask)
            combined_x = self.attention_core(q, k, v, attn_mask)
            outputs = self.out_conv(combined_x)

            return outputs


    def _split_heads(self, x):
        """
        Split channels (dimension 3) into multiple heads (dimension 1)
        input: (batch_size * ninp * inp_dim)
        output: (batch_size * n_heads * ninp * (inp_dim/n_heads))
        """
        old_shape = x.get_shape().dims
        dim_size = old_shape[-1]
        new_shape = old_shape[:-1] + [self.num_heads] + [dim_size // self.num_heads if dim_size else None]
        ret = tf.reshape(x, tf.concat([tf.shape(x)[:-1], [self.num_heads, tf.shape(x)[-1] // self.num_heads]], 0))
        ret.set_shape(new_shape)
        return tf.transpose(ret, [0, 2, 1, 3])  # [batch_size * n_heads * ninp * (hid_dim//n_heads)]

    def _combine_heads(self, x):
        """
        Inverse of split heads
        input: (batch_size * n_heads * ninp * (inp_dim/n_heads))
        out: (batch_size * ninp * inp_dim)
        """
        x = tf.transpose(x, [0, 2, 1, 3])
        old_shape = x.get_shape().dims
        a, b = old_shape[-2:]
        new_shape = old_shape[:-2] + [a * b if a and b else None]
        ret = tf.reshape(x, tf.concat([tf.shape(x)[:-2], [tf.shape(x)[-2] * tf.shape(x)[-1]]], 0))
        ret.set_shape(new_shape)
        return ret



class MultiHeadAttnConcrete(MultiHeadAttn):
    """
    Multihead scaled-dot-product attention with input/output transformations.
    This is the modification with scalar gates to each head, which enables head pruning introduced in https://arxiv.org/abs/1905.09418
    """

    def __init__(
            self, name, inp_size,
            key_depth, value_depth, output_depth,
            num_heads, attn_dropout, attn_value_dropout,
            kv_inp_size=None, _format='combined',
            gate_hp={'l0_penalty': 1.0},
    ):
        super().__init__(name, inp_size,
            key_depth, value_depth, output_depth,
            num_heads, attn_dropout, attn_value_dropout,
            kv_inp_size=kv_inp_size, _format=_format)

        self.gate_hp = gate_hp

        with tf.variable_scope(name):
            self.scope = tf.get_variable_scope()
            self.gate = ConcreteGate('gate', shape=[1, self.num_heads, 1, 1], **self.gate_hp)


    def __call__(self, query_inp, attn_mask, kv_inp=None, kv=None):
        """
        query_inp: [batch_size * n_q * inp_dim]
        attn_mask: [batch_size * 1 * n_q * n_kv]
        kv_inp: [batch_size * n_kv * inp_dim]
        -----------------------------------------------
        results: [batch_size * n_q * output_depth]
        """
        assert kv is None or kv_inp is None, "please only feed one of kv or kv_inp"
        with tf.name_scope(self.name) as scope:
            if kv_inp is not None or kv is not None:
                q = self.query_conv(query_inp)
                if kv is None:
                    kv = self.kv_conv(kv_inp)
                k, v = tf.split(kv, [self.key_depth, self.value_depth], axis=2)
            else:
                combined = self.combined_conv(query_inp)
                q, k, v = tf.split(combined, [self.key_depth, self.key_depth, self.value_depth], axis=2)
            q = self._split_heads(q)  # [batch_size * n_heads * n_q * (k_dim/n_heads)]
            k = self._split_heads(k)  # [batch_size * n_heads * n_kv * (k_dim/n_heads)]
            v = self._split_heads(v)  # [batch_size * n_heads * n_kv * (v_dim/n_heads)]

            key_depth_per_head = self.key_depth / self.num_heads
            q = q / math.sqrt(key_depth_per_head)

            # Dot-product attention
            # logits: (batch_size * n_heads * n_q * n_kv)
            attn_bias = MultiHeadAttn.ATTN_BIAS_VALUE * (1 - attn_mask)
            logits = tf.matmul(
                q,
                tf.transpose(k, perm=[0, 1, 3, 2])) + attn_bias
            weights = tf.nn.softmax(logits)

            tf.add_to_collection("AttnWeights", weights)

            tf.add_to_collection(lib.meta.ATTENTIONS, lib.meta.Attention(scope, weights, logits, attn_mask))

            if is_dropout_enabled():
                weights = dropout(weights, 1.0 - self.attn_dropout)
            x = tf.matmul(
                weights,  # [batch_size * n_heads * n_q * n_kv]
                v  # [batch_size * n_heads * n_kv * (v_deph/n_heads)]
            )
            # x: [batch, n_heads, n_q, (v_deph/n_heads)]

            # ========================  apply the gate  ========================
            gated_x = self.gate(x)

            tf.add_to_collection("CONCRETE", self.gate.get_sparsity_rate())
            tf.add_to_collection("GATEVALUES", self.gate.get_gates(False))
            # ==================================================================

            combined_x = self._combine_heads(gated_x)

            if is_dropout_enabled():
                combined_x = dropout(combined_x, 1.0 - self.attn_value_dropout)

            outputs = self.out_conv(combined_x)

            return outputs



class MultiHeadAttnFixedAliveHeads(MultiHeadAttn):
    """
    Multihead scaled-dot-product attention with input/output transformations.
    This is the modification with constant binary gates for each head,
    which specify which heads are present.
    Need to pass 'head_gate' parameter, which the list of num_heads values, one for each head.
    """

    def __init__(
            self, name, inp_size,
            key_depth, value_depth, output_depth,
            num_heads, attn_dropout, attn_value_dropout,
            kv_inp_size=None, _format='combined',
            head_gate=None,
    ):
        super().__init__(name, inp_size,
            key_depth, value_depth, output_depth,
            num_heads, attn_dropout, attn_value_dropout,
            kv_inp_size=kv_inp_size, _format=_format)

        assert head_gate is not None, "You must feed values for head gates"
        self.head_gate = head_gate

        with tf.variable_scope(name):
            self.scope = tf.get_variable_scope()
            self.gate = tf.constant(self.head_gate, dtype=tf.float32)[None, :, None, None]


    def __call__(self, query_inp, attn_mask, kv_inp=None, kv=None):
        """
        query_inp: [batch_size * n_q * inp_dim]
        attn_mask: [batch_size * 1 * n_q * n_kv]
        kv_inp: [batch_size * n_kv * inp_dim]
        -----------------------------------------------
        results: [batch_size * n_q * output_depth]
        """
        assert kv is None or kv_inp is None, "please only feed one of kv or kv_inp"
        with tf.name_scope(self.name) as scope:
            if kv_inp is not None or kv is not None:
                q = self.query_conv(query_inp)
                if kv is None:
                    kv = self.kv_conv(kv_inp)
                k, v = tf.split(kv, [self.key_depth, self.value_depth], axis=2)
            else:
                combined = self.combined_conv(query_inp)
                q, k, v = tf.split(combined, [self.key_depth, self.key_depth, self.value_depth], axis=2)
            q = self._split_heads(q)  # [batch_size * n_heads * n_q * (k_dim/n_heads)]
            k = self._split_heads(k)  # [batch_size * n_heads * n_kv * (k_dim/n_heads)]
            v = self._split_heads(v)  # [batch_size * n_heads * n_kv * (v_dim/n_heads)]

            key_depth_per_head = self.key_depth / self.num_heads
            q = q / math.sqrt(key_depth_per_head)

            # Dot-product attention
            # logits: (batch_size * n_heads * n_q * n_kv)
            attn_bias = MultiHeadAttn.ATTN_BIAS_VALUE * (1 - attn_mask)
            logits = tf.matmul(
                q,
                tf.transpose(k, perm=[0, 1, 3, 2])) + attn_bias
            weights = tf.nn.softmax(logits)

            tf.add_to_collection("AttnWeights", weights)

            tf.add_to_collection(lib.meta.ATTENTIONS, lib.meta.Attention(scope, weights, logits, attn_mask))

            if is_dropout_enabled():
                weights = dropout(weights, 1.0 - self.attn_dropout)
            x = tf.matmul(
                weights,  # [batch_size * n_heads * n_q * n_kv]
                v  # [batch_size * n_heads * n_kv * (v_deph/n_heads)]
            )
            # x: [batch, n_heads, n_q, (v_deph/n_heads)]

            # ========================  apply the gate  ========================
            gated_x = self.gate * x
            # ==================================================================

            combined_x = self._combine_heads(gated_x)

            if is_dropout_enabled():
                combined_x = dropout(combined_x, 1.0 - self.attn_value_dropout)

            outputs = self.out_conv(combined_x)

            return outputs

