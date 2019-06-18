import tensorflow as tf

###############################################################################
#
#  TensorFlow MPI operations
#
###############################################################################


def size(name=None):
    return tf.constant(1, name=name)


def rank(name=None):
    return tf.constant(0, name=name)


def local_rank(name=None):
    return tf.constant(0, name=name)


def init(name=None):
    return tf.no_op(name=name)


def finalize(name=None):
    return tf.no_op(name=name)


def allreduce(tensor, average=True, name=None):
    return tf.stop_gradient(tensor, name=name)  # Stop gradient propagation, as in distributed mode


def allgather(tensor, name=None):
    return tf.stop_gradient(tensor, name=name)


def broadcast(tensor, name=None):
    return tf.stop_gradient(tensor, name=name)


def broadcast_var(ref, allow_uninitialized=False, name=None):
    return ref


###############################################################################
#
#  Specific MPI operations on Python objects
#
###############################################################################


def broadcast_obj(obj, name=None):
    return obj


def gather_obj(obj, name=None):
    return [obj]


def scatter_obj(obj_array, name=None):
    return obj_array[0]


def allgather_obj(obj, name=None):
    return [obj]
