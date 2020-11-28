import tensorflow as tf
import math

import lib
from lib.ops.basic import is_dropout_enabled, dropout
from lib.ops import record_activations as rec
from .basic import Dense
from .lrp import LRP

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
        """
        Core math operations of multihead attention layer
        :param q, k, v: [batch_size, n_q or n_kv, dim per head]
        :param attn_head_mask: [batch_size, n_q, n_kv]
        """
        assert q.shape.ndims == 3 and attn_mask.shape.ndims == 3
        key_depth_per_head = tf.shape(q)[-1]
        q = q / tf.to_float(key_depth_per_head) ** 0.5

        attn_bias = self.ATTN_BIAS_VALUE * (1 - attn_mask)
        logits = tf.matmul(q, tf.transpose(k, perm=[0, 2, 1])) + attn_bias
        weights = tf.nn.softmax(logits)  # [batch_size, n_q, n_kv]

        tf.add_to_collection("AttnWeights", weights)
        tf.add_to_collection(lib.meta.ATTENTIONS, lib.meta.Attention(self.scope, weights, logits, attn_mask))

        if is_dropout_enabled():
            weights = dropout(weights, 1.0 - self.attn_dropout)
        x = tf.matmul(
            weights,  # [batch_size * n_q * n_kv]
            v  # [batch_size * n_kv * (v_deph/n_heads)]
        )

        if is_dropout_enabled():
            x = dropout(x, 1.0 - self.attn_value_dropout)
        return x

    def _attn_head_jacobian(self, q, k, v, attn_mask):
        """ same as lib.layers.lrp.jacobian(self.attention_core(q, k, v), [q, k, v]), but faster """
        # input shapes: (q, k, v) - [batch_size, n_q or n_kv, dim per head]
        # attn_head_mask: [batch_size, n_q, n_kv]
        assert q.shape.ndims == 3 and attn_mask.shape.ndims == 3
        key_depth_per_head = tf.shape(q)[-1]
        q = q / tf.to_float(key_depth_per_head) ** 0.5

        attn_bias = self.ATTN_BIAS_VALUE * (1 - attn_mask)
        logits = tf.matmul(q, tf.transpose(k, perm=[0, 2, 1])) + attn_bias
        weights = tf.nn.softmax(logits)  # [batch_size, n_q, n_kv]
        out = tf.matmul(weights, v)  # [batch_size, n_q, dim/n_heads]

        # compute jacobian w.r.t. values
        v_shape = tf.shape(v)
        batch_size, n_kv, dim_per_head = v_shape[0], v_shape[1], v_shape[2]

        diag_flat_weights = tf.einsum('ij,jqk->iqjk', tf.eye(tf.shape(weights)[0]), weights)  # [b, n_q, b, n_kv]
        flat_jac_v = diag_flat_weights[:, :, None, :, :, None] * tf.eye(dim_per_head)[None, None, :, None, None, :]
        # ^-- shape: [batch_size, n_q, dim/h, batch_size, n_kv, dim/h]

        jac_out_wrt_weights = tf.transpose(
            tf.tile(v[:, None], [1, tf.shape(out)[1], 1, 1]), [0, 1, 3, 2])
        # ^-- [batch_size, n_q, (dim), (n_kv)]
        softmax_jac = (weights[..., None] * tf.eye(tf.shape(weights)[-1])
                       - weights[..., None, :] * weights[..., :, None])  # <-- [batch_size, n_q, n_kv, n_kv]
        jac_out_wrt_logits = jac_out_wrt_weights @ softmax_jac  # [batch_size, n_q, (dim), (n_kv)]

        jac_out_wrt_k = jac_out_wrt_logits[..., None] * q[:, :, None, None, :]  # [b, (n_q, dim), (n_kv, dim)]

        # product axes:                    b  q  d  kv   d       b  q      d    kv d
        jac_out_wrt_q = jac_out_wrt_logits[:, :, :, :, None] * k[:, None, None, :, :]
        jac_out_wrt_q = tf.reduce_sum(jac_out_wrt_q, axis=3, keep_dims=True)
        jac_out_wrt_q = jac_out_wrt_q / tf.to_float(key_depth_per_head) ** 0.5
        jac_out_wrt_q = jac_out_wrt_q * tf.eye(tf.shape(jac_out_wrt_q)[1])[None, :, None, :, None]

        flat_jac_k = jac_out_wrt_k[..., None, :, :] * tf.eye(tf.shape(q)[0])[:, None, None, :, None, None]
        flat_jac_q = jac_out_wrt_q[..., None, :, :] * tf.eye(tf.shape(q)[0])[:, None, None, :, None, None]
        # final shape of flat_jac_{q, k}: [(batch_size, n_q, dim), (batch_size, n_kv, dim)]

        return flat_jac_q, flat_jac_k, flat_jac_v

    def _attn_head_jacobian_simple(self, q, k, v, attn_mask):
        jq, jk, jv = self._attn_head_jacobian(q, k, v, attn_mask)
        return tf.zeros_like(jq), tf.zeros_like(jk), jv

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

            q, k, v = map(self._split_heads, (q, k, v))  # [batch_size, n_heads, n_q or n_kv, k_dim / n_heads]
            q_flat, k_flat, v_flat = map(lambda x: tf.reshape(x, [-1, tf.shape(x)[2], tf.shape(x)[3]]), (q, k, v))
            # ^-- [batch_size, n_heads, n_q or n_kv, k_dim / n_heads]
            attn_mask_tiled = tf.tile(
                attn_mask, [tf.shape(q)[0] // tf.shape(attn_mask)[0],
                            tf.shape(q)[1] // tf.shape(attn_mask)[1], 1, 1])
            flat_attn_mask = tf.reshape(attn_mask_tiled, [-1, tf.shape(attn_mask)[2], tf.shape(attn_mask)[3]])
            # ^-- [(batch_size * n_heads), n_q, n_kv]

            rec.save_activations(q_flat=q_flat, k_flat=k_flat, v_flat=v_flat, flat_attn_mask=flat_attn_mask)
            x_flat = self.attention_core(q_flat, k_flat, v_flat, flat_attn_mask)
            combined_x = self._combine_heads(
                tf.reshape(x_flat, tf.concat([tf.shape(q)[:-1], tf.shape(v)[-1:]], axis=0)))
            outputs = self.out_conv(combined_x)

            return outputs

    def relprop(self, R):
        with tf.variable_scope(self.scope):
            assert rec.get_activation('kv') is None, "relprop through translatemodelfast is not implemented"
            R = self.out_conv.relprop(R)
            R_split = self._split_heads(R)

            # note: we apply relprop for each independent sample and head in order to avoid quadratic memory growth
            q_flat, k_flat, v_flat, flat_attn_mask = rec.get_activations('q_flat', 'k_flat', 'v_flat', 'flat_attn_mask')
            dim_per_head = tf.shape(v_flat)[-1]
            batch_size, n_heads, n_q = tf.shape(R_split)[0], tf.shape(R_split)[1], tf.shape(R_split)[2]
            n_kv = tf.shape(flat_attn_mask)[2]
            R_flat = tf.reshape(R_split, [-1, n_q, dim_per_head])
            # ^-- *_flat variables are of shape: [(batch * n_heads), n_q, dim per head]

            attn_jacobian = self._attn_head_jacobian_simple if LRP.consider_attn_constant else self._attn_head_jacobian

            flat_relevances = tf.map_fn(
                lambda i: LRP.relprop(
                    lambda q, k, v: self.attention_core(q, k, v, flat_attn_mask[i, None]),
                    R_flat[i, None], q_flat[i, None], k_flat[i, None], v_flat[i, None],
                    jacobians=attn_jacobian(q_flat[i, None], k_flat[i, None], v_flat[i, None], flat_attn_mask[i, None]),
                    batch_axes=(0,)),
                elems=tf.range(tf.shape(q_flat)[0]), dtype=[q_flat.dtype, k_flat.dtype, v_flat.dtype],
                parallel_iterations=1,  # note: more parallel_iterations causes slight speed-up at the cost of more RAM
            )
            Rq, Rk, Rv = [self._combine_heads(tf.reshape(rel_flat, [batch_size, n_heads, -1, dim_per_head]))
                          for rel_flat in flat_relevances]
            Rq, Rk, Rv = LRP.rescale(R, Rq, Rk, Rv, batch_axes=(0,))

            if rec.get_activation('is_combined'):
                Rqkv = tf.concat([Rq, Rk, Rv], axis=2)  # [batch, time, 3 * hid_size]
                Rinp = self.combined_conv.relprop(Rqkv)
                return Rinp
            else:
                Rkv = tf.concat([Rk, Rv], axis=2)  # [batch, time, 2 * hid_size]
                Rkvinp = self.kv_conv.relprop(Rkv)
                Rqinp = self.query_conv.relprop(Rq)
                return {'query_inp': Rqinp, 'kv_inp': Rkvinp}

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
