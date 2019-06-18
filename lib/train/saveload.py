# Saving Tensorflow variables to NPZ


import tensorflow as tf
import numpy as np


## ----------------------------------------------------------------------------
#                         Saving to / loading from NPZ


DO_NOT_SAVE = 'VARIABLES_DO_NOT_SAVE'


def save(filename, vars):
    """
    Save specified variables to an NPZ archive.
    """
    p = {}
    for var in vars:
        p[var.name] = var

    # 3. Evaluate all tensors at once
    keys = list(p.keys())
    values = tf.get_default_session().run([p[k] for k in keys])
    p = dict(zip(keys, values))

    # 3. Write.
    np.savez(filename, **p)


def load(filename, vars, batch_size=10):
    """
    Load NPZ archive into specified variables.

    If variable we want to load is not in NPZ, we ignore it.
    If NPZ has a value for a variable that is not in 'vars' list, we ignore it.
    """
    p = np.load(filename)
    ops = []
    feed_dict = {}

    with tf.variable_scope('load'):
        for var in vars:
            if var.name not in p.keys():
                continue

            # Create placeholder.
            placeholder = tf.placeholder(var.dtype)
            feed_dict[placeholder] = p[var.name]

            # Create assign op for normal vars.
            ops.append(tf.assign(var, placeholder, validate_shape=False).op)

    if ops:
        for ofs in range(0, len(ops), batch_size):
            tf.get_default_session().run(ops[ofs:ofs+batch_size], feed_dict)


def get_model_variables():
    """
    Return set of variables in tf.GraphKeys.MODEL_VARIABLES collection.
    """
    g = tf.get_default_graph()
    return set(g.get_collection(tf.GraphKeys.MODEL_VARIABLES))


def get_state_variables():
    """
    Return set of all tensorflow variables except:

        - Variables in tf.GraphKeys.MODEL_VARIABLES collection
        - Variables in DO_NOT_SAVE collection
    """
    g = tf.get_default_graph()
    vars = set(tf.global_variables())
    vars -= get_model_variables()
    vars -= set(g.get_collection(DO_NOT_SAVE))
    return vars


def initialize_uninitialized_variables():
    with tf.name_scope("initialize"):
        uninitialized_names = set(tf.get_default_session().run(tf.report_uninitialized_variables()))
        uninitialized_vars = []
        for var in tf.global_variables():
            if var.name[:-2].encode() in uninitialized_names:
                uninitialized_vars.append(var)

        tf.variables_initializer(uninitialized_vars).run()
