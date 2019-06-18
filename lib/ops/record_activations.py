from warnings import warn
from collections import defaultdict
from contextlib import contextmanager
import tensorflow as tf

# Idea: we need to store layer activations to do things like relevance propagation,
# let's build a single-use collection that one can store layer-wise activations in
# Here's how it should work:
# with record_activations() as saved_activations:
#    y = model(x)              # saves activations in... saved_activations
#    x_rel = model.relprop(y)  # uses activations stored on forward pass
#
# print('btw, activation tensors are', activations)
# note: why not just use tf collections? because they are global and you can never be sure
# what's left in there since previous run

# this will be a dictionary: { layer name -> a dict of saved activations }
RECORDED_ACTIVATIONS = None
WARN_IF_NO_COLLECTION = False


@contextmanager
def recording_activations(existing_state_dict=None, subscope_key=None):
    """ A special context that allows you to store any forward pass activations """
    assert isinstance(existing_state_dict, (dict, type(None)))
    global RECORDED_ACTIVATIONS
    prev_collection = RECORDED_ACTIVATIONS
    RECORDED_ACTIVATIONS = existing_state_dict or defaultdict(dict)
    if subscope_key:
        assert is_recorded() and existing_state_dict is None
        prev_collection[subscope_key] = RECORDED_ACTIVATIONS

    try:
        yield RECORDED_ACTIVATIONS
    finally:
        RECORDED_ACTIVATIONS = prev_collection


@contextmanager
def do_not_record():
    """ Temporarily disables recording activations within context """
    global RECORDED_ACTIVATIONS
    prev_collection = RECORDED_ACTIVATIONS
    RECORDED_ACTIVATIONS = None
    try:
        yield
    finally:
        RECORDED_ACTIVATIONS = prev_collection


def is_recorded():
    return RECORDED_ACTIVATIONS is not None


def save_activation(key, value, scope=None, overwrite=False):
    """ Saves value in current recorded activations (if it exists) under current name scope """
    scope = scope or tf.get_variable_scope().name or tf.contrib.framework.get_name_scope()
    if is_recorded():
        if scope in RECORDED_ACTIVATIONS and key in RECORDED_ACTIVATIONS[scope] and not overwrite:
            raise ValueError('Recorded activations already contain key "{}" for scope "{}". '
                             'Make sure you run your network only once inside recording_activations context. '
                             'If a layer is called multiple times, make sure each call happens in a separate '
                             ' tf.name_scope .'.format(key, scope))

        RECORDED_ACTIVATIONS[scope][key] = value
    elif WARN_IF_NO_COLLECTION:
        warn('Tried to save under key "{}" in scope "{}" without recording_activations context. '
             'As the fox says, the context is important'.format(key, scope))


def save_activations(**kwargs):
    """ convenience function to save multiple activations. see save_activation """
    scope, overwrite = kwargs.pop('scope', None), kwargs.pop('overwrite', False)
    assert isinstance(scope, (str, type(None)))
    assert isinstance(overwrite, bool)
    for key, value in kwargs.items():
        save_activation(key, value, scope=scope, overwrite=overwrite)


def get_activation(key, scope=None):
    """ gets one activation from current scope or freaks out if there isn't any """
    scope = scope or tf.get_variable_scope().name or tf.contrib.framework.get_name_scope()
    assert is_recorded(), "can't get activations if used outside recording_activations context."
    assert scope in RECORDED_ACTIVATIONS, 'no saved activations in scope "{}". Is scope name correct?'.format(scope)
    assert key in RECORDED_ACTIVATIONS[scope], 'no saved activation for "{}" in scope "{}". Existing keys: {}'.format(
        key, scope, list(RECORDED_ACTIVATIONS[scope].keys())
    )
    return RECORDED_ACTIVATIONS[scope][key]


def get_activations(*keys, scope=None):
    """ convenience function to get multiple activations from current scope, see get_activation """
    return [get_activation(key, scope=scope) for key in keys]
