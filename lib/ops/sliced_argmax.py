import numpy as np
import tensorflow as tf


def hypo_to_batch_index(n_hypos, slices):
    """
    Computes index in batch (input sequence index) for each hypothesis given slices.
    :param n_hypos: number of hypotheses (tf int scalar)
    :param slices: indices of first hypo for each input in batch
    It should guaranteed that
     - slices[0]==0 (first hypothesis starts at index 0), otherwise output[:slices[0]] will be -1
     - if batch[i] is terminated, then batch[i]==batch[i+1]
    """
    is_next_sent_at_t = tf.bincount(slices, minlength=n_hypos, maxlength=n_hypos)
    hypo_to_index = tf.cumsum(is_next_sent_at_t) - 1
    return hypo_to_index


def sliced_argmax_naive(logits, slices, k):
    """
    Computes top-k of values in each slice.
    :param values: matrix of shape [m,n]
    :param slices: vector of shape [m] containing start indices for each slice.
    :param k: take this many elements with largest values from each slice
    :returns: batch_scores,batch_indices:
        - batch_scores[m,k] - top-beam_size values from logP corresponding to
        - batch_indices[m,k] - indices of batch_scores in each respective slice (first value in each slice has index 0!)

    For any slice contains less than k elements, batch_scores would be padded with -inf, batch_indices - with -1
    If values.shape[1] != 1, batch_indices will still be 1-dimensional, satisfying the following property:
        - batch_scores,batch_indices = sliced_argmax(values,slices,k)
        - start, end = slices[i], slices[i+1]
        - tf.equals(batch_scores == tf.reshape(values[start:end,:],[-1])[batch_indices])  #this is True for all indices

    Examples
    --------
    >>> logp = tf.constant(np.array([[1, 2,   3, 4, 5,   6],
                                     [6, 5,   4, 3, 2,   1]],'float32').T)
    >>> slices = tf.constant([0,2,5])
    >>> best_scores, best_indices = sliced_argmax(logp,slices,tf.constant(4))
    >>> print('scores:\n%s\nindices:\n%s'%(best_scores.eval(), best_indices.eval()))
    scores:
    [[  6.   5.   2.   1.]
     [  5.   4.   4.   3.]
     [  6.   1. -inf -inf]]
    indices:
    [[ 1  3  2  0]
     [ 4  1  2  3]
     [ 0  1 -1 -1]]
    """

    assert logits.shape.ndims == 2, "logits must be [batch*beam, num_tokens]"
    assert slices.shape.ndims == 1, "slices must be 1d indices"
    n_slices, n_hypos, voc_size = tf.shape(slices)[0], tf.shape(logits)[0], tf.shape(logits)[1]
    slices_incl = tf.concat([slices, [n_hypos]], axis=0)
    offsets = slices_incl[1:] - slices_incl[:-1]
    slice_indices = hypo_to_batch_index(n_hypos, slices)  # [n_hypos], index of slice the value belongs to

    # step 1: flatten logits[n_hypos, voc_size] into [n_slices, max_slice_length * voc_size]
    # by putting all logits within slice on the same row and padding with -inf
    flat_shape = [n_slices, (tf.reduce_max(offsets)) * voc_size]
    flat_row_index = tf.reshape(tf.tile(slice_indices[:, None], [1, voc_size]), [-1])
    flat_col_index = tf.range(n_hypos * voc_size) - tf.gather(slices_incl * voc_size, flat_row_index)
    flat_index_2d = tf.stack([flat_row_index, flat_col_index], axis=1)
    mask = tf.less(tf.range(flat_shape[1]), (offsets * voc_size)[:, None])
    flat_logits = tf.where(mask,
                           tf.scatter_nd(flat_index_2d, tf.reshape(logits, [-1]), flat_shape),
                           tf.fill(flat_shape, -float('inf'))
                           )  # shape: [n_slices, max_slice_length * voc_size]

    flat_indices = tf.where(mask,
                            tf.scatter_nd(flat_index_2d, flat_col_index, flat_shape),
                            tf.fill(flat_shape, -1)
                            )  # shape: [n_slices, max_slice_length * voc_size]

    # step 2: top-k for each slice and gather respectrive indices
    sliced_top_k = tf.nn.top_k(flat_logits, k=k)
    original_values = sliced_top_k.values

    original_indices_flat = tf.gather_nd(flat_indices,
                                         tf.stack([tf.range(n_slices * k) // k,
                                                   tf.reshape(sliced_top_k.indices, [-1])], axis=1))
    original_indices = tf.reshape(original_indices_flat, tf.shape(original_values))

    # set shapes
    out_shape = (logits.shape[0], k if isinstance(k, int) else None)
    original_values.set_shape(out_shape)
    original_indices.set_shape(out_shape)
    return original_values, original_indices


def sliced_argmax(logits, slices, k, staged=None):
    """
    Computes top-k of values in each slice.
    :param values: matrix of shape [m,n]
    :param slices: vector of shape [m] containing start indices for each slice.
    :param k: take this many elements with largest values from each slice
    :param staged: if True, computes sliced argmax in two stages:
                (1) select top-k for each row and
                (2) global top-k among all rows in slice
            if False, runs second stage only
            if None (default), defaults to True unless logits.shape[1] / k < 10
    :returns: batch_scores,batch_indices:
        - batch_scores[m,k] - top-beam_size values from logP corresponding to
        - batch_indices[m,k] - indices of batch_scores in each respective slice (first value in each slice has index 0!)

    For any slice contains less than k elements, batch_scores would be padded with -inf, batch_indices - with -1
    If values.shape[1] != 1, batch_indices will still be 1-dimensional, satisfying the following property:
        - batch_scores,batch_indices = sliced_argmax(values,slices,k)
        - start, end = slices[i], slices[i+1]
        - tf.equals(batch_scores == tf.reshape(values[start:end,:],[-1])[batch_indices])  #this is True for all indices

    Examples
    --------
    >>> logp = tf.constant(np.array([[1, 2,   3, 4, 5,   6],
                                     [6, 5,   4, 3, 2,   1]],'float32').T)
    >>> slices = tf.constant([0,2,5])
    >>> best_scores, best_indices = sliced_argmax(logp,slices,tf.constant(4))
    >>> print('scores:\n%s\nindices:\n%s'%(best_scores.eval(), best_indices.eval()))
    scores:
    [[  6.   5.   2.   1.]
     [  5.   4.   4.   3.]
     [  6.   1. -inf -inf]]
    indices:
    [[ 1  3  2  0]
     [ 4  1  2  3]
     [ 0  1 -1 -1]]
    """

    assert logits.shape.ndims == 2, "logits must be [batch*beam, num_tokens]"
    assert slices.shape.ndims == 1, "slices must be 1d indices"
    if staged is None:
        staged = (logits.shape[1].value is None) or (float(logits.shape[1].value) / k >= 10.0)

    if staged:
        # two-step process: (1) select top-k for each row and (2) global top-k among all rows in slice
        # this version is slightly slower but a lot more memory-efficient
        logits_topk = tf.nn.top_k(logits, k=k)  # [n_hypos, k]
        best_values, best_indices_in_top = sliced_argmax_naive(logits_topk.values, slices, k=k)

        best_hypo_ix = tf.where(tf.not_equal(best_indices_in_top, -1),
                                best_indices_in_top // k + slices[:, None],
                                best_indices_in_top)

        best_token_ix_in_top = tf.where(tf.not_equal(best_indices_in_top, -1),
                                        best_indices_in_top % k,
                                        best_indices_in_top)

        best_token_indices_original = tf.gather_nd(
            logits_topk.indices,
            tf.maximum(0, tf.reshape(tf.stack([best_hypo_ix, best_token_ix_in_top], axis=-1), [-1, 2]))
        )
        best_token_indices_original = tf.where(tf.not_equal(tf.reshape(best_hypo_ix, [-1]), -1),
                                               best_token_indices_original,
                                               tf.fill(tf.shape(best_token_indices_original), -1))

        best_token_indices_original = tf.reshape(best_token_indices_original,
                                                 tf.shape(best_token_ix_in_top))
        best_hypo_ix_within_slice = tf.where(
            tf.not_equal(best_indices_in_top, -1),
            best_indices_in_top // k,
            tf.zeros_like(best_indices_in_top, dtype=best_indices_in_top.dtype))
            #  ^-- use 0 cuz best_token_indices_original is already -1 and they are added

        best_indices_original = best_token_indices_original + best_hypo_ix_within_slice * tf.shape(logits)[1]
        return best_values, best_indices_original
    else:
        return sliced_argmax_naive(logits, slices, k)
