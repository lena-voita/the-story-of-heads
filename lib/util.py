# Utilities
import numpy as np
import tensorflow as tf

import importlib
import contextlib


## ----------------------------------------------------------------------------
#                        Nested dicts / lists / tuples


def nested_compare(t, u):
    """
    Return whether nested structure of t1 and t2 matches.
    """
    if isinstance(t, (list, tuple)):
        if not isinstance(u, type(t)):
            return False
        if len(t) != len(u):
            return False
        for a, b in zip(t, u):
            if not nested_compare(a, b):
                return False
        return True

    if isinstance(t, dict):
        if not isinstance(u, dict):
            return False
        if set(t.keys()) != set(u.keys()):
            return False
        for k in t:
            if not nested_compare(t[k], u[k]):
                return False
        return True

    else:
        return True


def nested_flatten(t):
    """
    Turn nested list/tuple/dict into a flat iterator.
    """
    if isinstance(t, (list, tuple)):
        for x in t:
            yield from nested_flatten(x)
    elif isinstance(t, dict):
        for k, v in sorted(t.items()):
            yield from nested_flatten(v)
    else:
        yield t


def nested_pack(flat, structure):
    return _nested_pack(iter(flat), structure)


def _nested_pack(flat_iter, structure):
    if is_namedtuple(structure):
        return type(structure)(*[
            _nested_pack(flat_iter, x)
            for x in structure]
            )
    if isinstance(structure, (list, tuple)):
        return type(structure)(
            _nested_pack(flat_iter, x)
            for x in structure
            )
    elif isinstance(structure, dict):
        return {
            k: _nested_pack(flat_iter, v)
            for k, v in sorted(structure.items())
            }
    else:
        return next(flat_iter)


def is_namedtuple(x):
    """Checks if x is a namedtuple instance. Taken from https://stackoverflow.com/a/2166841 ."""
    t = type(x)
    b = t.__bases__
    if len(b) != 1 or b[0] != tuple: return False
    f = getattr(t, '_fields', None)
    if not isinstance(f, tuple): return False
    return all(type(n)==str for n in f)


def nested_map(fn, *t):
    # Check arguments.
    if not t:
        raise ValueError('Expected 2+ arguments, got 1')
    for i in range(1, len(t)):
        if not nested_compare(t[0], t[i]):
            msg = 'Nested structure of %r and %r differs'
            raise ValueError(msg % (t[0], t[i]))

    # Map.
    flat = map(nested_flatten, t)
    return nested_pack(map(fn, *flat), t[0])


## ----------------------------------------------------------------------------
#                         Variables and initialization


def with_shape(var, shape):
    tensor = var.value()
    tensor.set_shape(shape)
    return tensor


def orthogonal(shape):
    """
    Generate a random orthogonal matrix (with ortogonal columns) via SVD of
    random matrix.

    Raises ValueError when shape.ndim != 2 or when shape[0] < shape[1].

    'shape' is np.array.
    """
    if shape.ndim != 2:
        msg = 'Can generate only orthogonal matrices, not %id tensors'
        raise ValueError(msg % shape.ndim)
    if shape[0] < shape[1]:
        msg = ('Cannot generate %i orthogonal columns, each of which consists'
               'of %i numbers (guess why)')
        raise ValueError(msg % (shape[1], shape[0]))
    a = np.random.normal(0.0, 1.0, shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    return u


def orthogonal_initializer(scale=1.0):
    def initializer(shape, dtype=tf.float32, partition_info=None):
        return tf.constant(orthogonal(shape) * scale, dtype)
    return initializer


## ----------------------------------------------------------------------------
#                               Miscellaneous


def load_class(full_name):
    name_parts = full_name.split('.')

    module_name = '.'.join(name_parts[:-1])
    class_name = name_parts[-1]

    return getattr(importlib.import_module(module_name), class_name)


def merge_dicts(a, b):
    res = a.copy()
    res.update(b)
    return res


def is_scalar(var):
    """ checks if var is not scalar. Works for list, np.array, tf.tensor and many similar classes """
    return len(np.shape(var)) == 0


@contextlib.contextmanager
def nop_ctx():
    yield


def merge_summaries(inputs, collections=None, name=None):
    # Wrapper correctly working with inputs = []
    if len(inputs) == 0:
        # We should return simple tf operation that returns empty bytes
        return tf.identity(b'')
    return tf.summary.merge(inputs, collections, name)
