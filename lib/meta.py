import tensorflow as tf
import sys

from collections import namedtuple
from contextlib import contextmanager

## Collection keys

# Collection of tensors representing layer activations in network
ACTIVATIONS = tf.GraphKeys.ACTIVATIONS

# Collection of Attention objects
ATTENTIONS = "attentions"
SUMMARIES_ZOO = "summaries_zoo"
PARAMS_SUMMARIES = "params_summaries"


Attention = namedtuple('Attention', ['name', 'weights', 'logits', 'mask'])


def get_indexed_collection(coll, scope, root_scope=None):
    if root_scope is None:
        root_scope = tf.contrib.framework.get_name_scope()

    full_scope = root_scope + '/' + scope

    def normalize_name(n):
        n = n[len(full_scope)+1:]
        if n.endswith(':0'):
            n = n[:-2]
        if n.endswith('/'):
            n = n[:-1]
        return n

    return dict((normalize_name(t.name), t) for t in tf.get_collection(coll, full_scope + '/.*'))


@contextmanager
def lock_collections(collections):
    collection_states = [tf.get_collection(coll) for coll in collections]
    yield
    for coll, old_coll_state in zip(collections, collection_states):
        new_coll_state = tf.get_collection_ref(coll)
        if old_coll_state != new_coll_state:
            print("! Changes in collection %s will be ignored!" % coll, flush=True, file=sys.stderr)
            new_coll_state[:] = old_coll_state  # Replace collection state with old one
