# Basic TF operations
import threading
from contextlib import contextmanager

import tensorflow as tf
import hashlib
from copy import copy


def get_seed_from_name(name):
    full_name = '/'.join([tf.get_variable_scope().name, name])
    return int(hashlib.md5(full_name.encode()).hexdigest()[:8], 16)


def default_initializer(seed, dtype):
    scope_initializer = tf.get_variable_scope().initializer
    if scope_initializer is not None:
        return scope_initializer
    try:
        return tf.initializers.glorot_uniform(seed, dtype)
    except:
        return tf.glorot_uniform_initializer(seed, dtype)


def get_model_variable(name, **kwargs):
    """ Get variable from MODEL_VARIABLES collection with initializer seeded from its name, not id """

    if kwargs.get('initializer') is None:
        kwargs['initializer'] = default_initializer(seed=get_seed_from_name(name), dtype=kwargs.get('dtype', tf.float32))
    elif hasattr(kwargs['initializer'], 'seed') and kwargs['initializer'].seed is None:
        kwargs['initializer'] = copy(kwargs['initializer'])
        kwargs['initializer'].seed = get_seed_from_name(name)

    return tf.contrib.framework.model_variable(name, **kwargs)


def dot(x, y):
    """
    x: [..., a]
    y: [a, ...]
    -------------
    Ret: [..., ...]
    """
    x_ndim = x.get_shape().ndims
    y_ndim = y.get_shape().ndims
    etc_x = tf.slice(tf.shape(x), [0], [x_ndim-1])
    etc_y = tf.slice(tf.shape(y), [1], [-1])
    a = tf.shape(y)[0]

    # Reshape forth.
    if x_ndim != 2:
        x = tf.reshape(x, [-1, a])
    if y_ndim != 2:
        y = tf.reshape(y, [a, -1])

    # Compute
    ret = tf.matmul(x, y)

    # Reshape back.
    if x_ndim != 2 or y_ndim != 2:
        ret = tf.reshape(ret, tf.concat([etc_x, etc_y], 0))

    return ret


def sequence_mask(lengths, dtype, maxlen=None):
    """
    WARNING: THis func produces Time-major tensor
    lengths: [batch_size]
    -------
    out: [maxlen, batch_size]
    """
    lengths = tf.cast(lengths, tf.int32)
    if maxlen is not None:
        maxlen = tf.cast(maxlen, tf.int32)
    return tf.transpose(tf.sequence_mask(lengths, dtype=dtype, maxlen=maxlen))


def infer_length(seq, eos=1, time_major=False):
    """
    compute length given output indices and eos code
    :param seq: tf matrix [time,batch] if time_major else [batch,time]
    :param eos: integer index of end-of-sentence token
    :returns: lengths, int32 vector of [batch_size]
    """
    axis = 0 if time_major else 1
    is_eos = tf.cast(tf.equal(seq, eos), 'int32')
    count_eos = tf.cumsum(is_eos, axis=axis, exclusive=True)
    lengths = tf.reduce_sum(tf.cast(tf.equal(count_eos, 0), 'int32'), axis=axis)
    return lengths


def infer_mask(seq, eos=1, time_major=False, dtype=tf.bool):
    """
    compute mask
    :param seq: tf matrix [time,batch] if time_major else [batch,time]
    :param eos: integer index of end-of-sentence token
    :returns: mask, matrix of same shape as seq and of given dtype (bool by default)
    """
    lengths = infer_length(seq, eos=eos, time_major=time_major)
    mask_fn = sequence_mask if time_major else tf.sequence_mask
    maxlen = tf.shape(seq)[0 if time_major else 1]
    return mask_fn(lengths, dtype=dtype, maxlen=maxlen)


def dropout(x, keep_prob, *args, **kwargs):
    """This is a hack to save memory if there is no dropout"""
    if keep_prob >= 1:
        return x
    return tf.nn.dropout(x, keep_prob, *args, **kwargs)


def group(*ops):
    """
    Like tf.group(), but returns tf.constant(0) instead of tf.no_op(),
    which makes it suitable for use in tf.cond().
    """
    with tf.control_dependencies(ops):
        return tf.constant(0)


def select_values_over_last_axis(values, indices):
    """
    Auxiliary function to select logits corresponding to chosen tokens.
    :param values: logits for all actions: float32[batch,tick,action]
    :param indices: action ids int32[batch,tick]
    :returns: values selected for the given actions: float[batch,tick]
    """
    assert values.shape.ndims == 3 and indices.shape.ndims == 2
    batch_size, seq_len = tf.shape(indices)[0], tf.shape(indices)[1]

    time_i, batch_i = tf.meshgrid(tf.range(0, seq_len, dtype=indices.dtype),
                                  tf.range(0, batch_size, dtype=indices.dtype))

    indices_nd = tf.stack([batch_i, time_i, indices], axis=-1)

    return tf.gather_nd(values, indices_nd)


def nop(x):
    return x


def kl_divergence_with_logits(p_logits, q_logits):
    return tf.reduce_sum(tf.nn.softmax(p_logits) * (tf.nn.log_softmax(p_logits) - tf.nn.log_softmax(q_logits)), axis=-1)


_tls = threading.local()


def is_dropout_enabled():
    if not hasattr(_tls, 'dropout_enabled'):
        _tls.dropout_enabled = True
    return _tls.dropout_enabled


@contextmanager
def dropout_scope(enabled):
    was_enabled = is_dropout_enabled()
    _tls.dropout_enabled = enabled
    try:
        yield
    finally:
        _tls.dropout_enabled = was_enabled