import tensorflow as tf
import horovod.tensorflow as hvd
import pickle
import os
import threading


###############################################################################
#
#  Horovod MPI operations
#
###############################################################################


def size(name=None):
    return tf.constant(int(os.getenv('OMPI_COMM_WORLD_SIZE', 1)), name=name, dtype=tf.int32)


def rank(name=None):
    return tf.constant(int(os.getenv('OMPI_COMM_WORLD_RANK', 0)), name=name, dtype=tf.int32)


def local_rank(name=None):
    return tf.constant(int(os.getenv('OMPI_COMM_WORLD_LOCAL_RANK', 0)), name=name, dtype=tf.int32)


def init(name=None):
    hvd.init()
    return tf.no_op(name=name)


def finalize(name=None):
    return tf.no_op(name=name)


def allreduce(tensor, average=True, name=None):
    return hvd.allreduce(tensor, average=average)


def allgather(tensor, name=None):
    return hvd.allgather(tensor)


def broadcast(tensor, name=None):
    return hvd.broadcast(tensor, root_rank=0)


def broadcast_var(ref, allow_uninitialized=False, name=None):
    if allow_uninitialized:
        raise RuntimeError("allow_uninitialized is not supported in Horovod implementation")
    return tf.assign(ref, broadcast(ref))


###############################################################################
#
#  Specific MPI operations on Python objects
#
###############################################################################


def broadcast_obj(obj, name=None):
    if name is None:
        name = 'broadcast_obj'
    return allgather_obj(obj, name)[0]


def gather_obj(obj, name=None):
    if name is None:
        name = 'gather_obj'

    res = allgather_obj(obj, name)
    if int(os.getenv('OMPI_COMM_WORLD_RANK', 0)) == 0:
        return res
    else:
        return None


def scatter_obj(inps, name=None):
    if name is None:
        name = 'scatter_obj'

    if int(os.getenv('OMPI_COMM_WORLD_RANK', 0)) == 0:
        assert len(inps) == int(os.getenv('OMPI_COMM_WORLD_SIZE', 1))
    else:
        inps = None

    outs = allgather_obj(inps, name)
    return outs[0][int(os.getenv('OMPI_COMM_WORLD_RANK', 0))]


def allgather_obj(obj, name=None):
    if name is None:
        name = 'allgather_obj'

    encoded = _encode_obj(obj)
    encoded_size = len(encoded)

    graph_ops = _get_graph_ops(name)

    sizes, encoded_res = tf.get_default_session().run([graph_ops.allgather_obj_size_result, graph_ops.allgather_obj_result], feed_dict={
        graph_ops.allgather_obj_size_inp: [encoded_size],
        graph_ops.allgather_obj_inp: encoded
    })

    res = []
    pos = 0
    for sz in sizes:
        res.append(_decode_obj(encoded_res[pos:pos+sz]))
        pos += sz
    return res


## Implementation details

class _GraphOps:
    def __init__(self, name):
        self.name = name

        with tf.name_scope("horovod_python_ops/" + name):
            self.allgather_obj_size_inp = tf.placeholder(name="allgather_obj_size", dtype=tf.int32, shape=[None])
            self.allgather_obj_inp = tf.placeholder(name="allgather_obj", dtype=tf.uint8, shape=[None])

            self.allgather_obj_size_result = hvd.allgather(self.allgather_obj_size_inp)
            self.allgather_obj_result = hvd.allgather(self.allgather_obj_inp)


_graph_ops_collection = "HOROVOD_GRAPH_OPS"
_graph_ops_lock = threading.Lock()


def _encode_obj(obj):
    return list(pickle.dumps(obj))


def _decode_obj(data):
    return pickle.loads(bytes(data))


def _get_graph_ops(name):
    """
    Returns lazy-initialized hash of graph operations required to implement allgather_obj/scatter_obj.
    These operations stored in graph collection to avoid binding parallelism to specific graph
    """

    found = tf.get_collection(_graph_ops_collection, name)
    if len(found) > 0:
        return found[0]

    with _graph_ops_lock:
        ops = _GraphOps(name)
        tf.add_to_collection(_graph_ops_collection, ops)
        return ops
