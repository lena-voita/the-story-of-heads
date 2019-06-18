import tensorflow as tf
from ...ops.basic import select_values_over_last_axis


def append_counters_accuracy(counters, logits, out, out_len):
    with tf.variable_scope("summary_accuracy"):
        predictions = tf.argmax(logits, axis=2)
        acc_values = predictions2accuracy(predictions, out, out_len)
        acc_top5_values = logits2accuracy_top_k(logits, out, out_len, k=5)
        acc_per_seq_values = predictions2accuracy_per_sequence(predictions, out, out_len)

        node = dict(
            accuracy=tf.reduce_sum(acc_values),
            accuracy_top5=tf.reduce_sum(acc_top5_values),
            accuracy_per_sequence=tf.reduce_sum(acc_per_seq_values),
            out_len=tf.to_float(tf.reduce_sum(out_len)),
            seqs=tf.to_float(tf.shape(out_len)[0]),
        )

        _append_counters(counters, "summarize_accuracy", node)


def append_counters_common_metrics(counters, logits, out, out_len, is_train):
    append_counters_accuracy(counters, logits, out, out_len)


def append_counters_xent(counters, xent_values, out_len):
    with tf.variable_scope("summary_xent"):
        node = dict(
            xent=tf.reduce_sum(xent_values),
            out_len=tf.to_float(tf.reduce_sum(out_len)),
        )
        _append_counters(counters, "summarize_xent", node)


def append_counters_io(counters, inp, out, inp_len, out_len):
    with tf.variable_scope("summary_io"):
        node = dict(
            batch_size=tf.to_float(tf.shape(inp))[0],
            inp_len=tf.to_float(tf.reduce_sum(inp_len)),
            out_len=tf.to_float(tf.reduce_sum(out_len)),
            ninp=tf.to_float(tf.shape(inp)[1]),
            nout=tf.to_float(tf.shape(out)[1]),
        )
        _append_counters(counters, "summarize_io", node)


def summarize_accuracy(counters, prefix):
    node = counters['summarize_accuracy']
    summaries = [
        tf.summary.scalar("%s_metrics/Acc" % prefix, tf.reduce_sum(node['accuracy']) / tf.reduce_sum(node['out_len'])),
        tf.summary.scalar("%s_metrics/AccTop5" % prefix, tf.reduce_sum(node['accuracy_top5']) / tf.reduce_sum(node['out_len'])),
        tf.summary.scalar("%s_metrics/AccPerSeq" % prefix, tf.reduce_sum(node['accuracy_per_sequence']) / tf.reduce_sum(node['seqs'])),
    ]
    return summaries


def summarize_common_metrics(counters, prefix):
    return summarize_accuracy(counters, prefix)


def summarize_xent(counters, prefix):
    node = counters['summarize_xent']
    return [
        tf.summary.scalar("%s_metrics/Xent" % prefix, tf.reduce_sum(node['xent']) / tf.reduce_sum(node['out_len'])),
    ]


def summarize_io(counters, prefix):
    node = counters['summarize_io']
    return [
        tf.summary.scalar("%s_IO/BatchSize" % prefix, tf.reduce_sum(node['batch_size'])),
        tf.summary.scalar("%s_IO/InpLenAvg" % prefix, tf.reduce_sum(node['inp_len']) / tf.reduce_sum(node['batch_size'])),
        tf.summary.scalar("%s_IO/OutLenAvg" % prefix, tf.reduce_sum(node['out_len']) / tf.reduce_sum(node['batch_size'])),
        tf.summary.scalar("%s_IO/InpLenSum" % prefix, tf.reduce_sum(node['inp_len'])),
        tf.summary.scalar("%s_IO/OutLenSum" % prefix, tf.reduce_sum(node['out_len'])),

        tf.summary.scalar(
            "%s_IO/InpNoPadRate" % prefix,
            tf.reduce_sum(node['inp_len']) / tf.reduce_sum(node['ninp'] * node['batch_size'])),
        tf.summary.scalar(
            "%s_IO/OutNoPadRate" % prefix,
            tf.reduce_sum(node['out_len']) / tf.reduce_sum(node['nout'] * node['batch_size'])),
    ]


def _append_counters(counters, key, value):
    if isinstance(counters, dict):
        if key in counters:
            raise Exception('Duplicate key "{}" in counters'.format(key))
        counters[key] = value
    else:
        raise Exception('Unexpected type: {}. Counters should be dict'.format(counters.__class__.__name__))


def logits2accuracy(logits, out, out_len, dtype=tf.float32):
    """
    logits : [batch_size * nout * voc_size]
    out : [batch_size * nout]
    out_len: [batch_size]

    results: [batch_size * nout]
    """
    predictions = tf.argmax(logits, axis=2)
    return predictions2accuracy(predictions, out, out_len, dtype=dtype)


def predictions2accuracy(predictions, out, out_len, dtype=tf.float32):
    """
    predictions: [batch_size * nout]
    out : [batch_size * nout]
    out_len: [batch_size]

    results: [batch_size * nout]
    """
    out_equals = tf.equal(tf.cast(predictions, dtype=out.dtype), out)
    out_mask = tf.sequence_mask(out_len, dtype=dtype, maxlen=tf.shape(out)[1])
    acc_values = tf.cast(out_equals, dtype=dtype) * out_mask

    return acc_values


def logits2accuracy_top_k(logits, out, out_len, k, dtype=tf.float32):
    """
    logits: [batch_size * nout * ntokens]
    out : [batch_size * nout]
    out_len: [batch_size]

    results: [batch_size * nout]
    """
    out_logits = select_values_over_last_axis(logits, tf.to_int32(out))
    out_logits = tf.expand_dims(out_logits, axis=-1)

    greater_mask = tf.greater(logits, out_logits)
    greater_ranks = tf.reduce_sum(tf.to_int32(greater_mask), axis=-1)
    hit_mask = greater_ranks < k
    out_mask = tf.sequence_mask(out_len, dtype=dtype, maxlen=tf.shape(out)[1])
    acc_values = tf.to_float(hit_mask) * out_mask

    return acc_values


def predictions2accuracy_per_sequence(predictions, out, out_len, dtype=tf.float32):
    """
    predictions: [batch_size * nout]
    out: [batch_size * nout]
    out_len: [batch_size]

    results: [batch_size]
    """
    not_correct = tf.not_equal(tf.cast(predictions, dtype=out.dtype), out)
    out_mask = tf.sequence_mask(out_len, dtype=dtype, maxlen=tf.shape(out)[1])
    correct_seq = 1.0 - tf.minimum(1.0, tf.reduce_sum(tf.cast(not_correct, dtype=dtype) * out_mask, axis=1))
    return tf.cast(correct_seq, dtype=dtype)





