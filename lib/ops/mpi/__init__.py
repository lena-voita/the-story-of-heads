# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# pylint: disable=g-short-docstring-punctuation
"""## Communicating Between Processes with MPI

TensorFlow natively provides inter-device communication through send and
receive ops and inter-node communication through Distributed TensorFlow, based
on the same send and receive abstractions. On HPC clusters where Infiniband or
other high-speed node interconnects are available, these can end up being
insufficient for synchronous data-parallel training (without asynchronous
gradient descent). This module implements a variety of MPI ops which can take
advantage of hardware-specific MPI libraries for efficient communication.

In order to use this module, TensorFlow must be built with an MPI library,
which can be provided to the `./configure` script at build time. As a user of
TensorFlow, you will need to build TensorFlow yourself to select the MPI
library to use; to do so, follow the [instructions for building TensorFlow from
source](https://www.tensorflow.org/get_started/os_setup#installing_from_sources).

### Utility Ops

In addition to reductions and gathers, this module provides utility operations
for detecting the running MPI configuration.

Example:

```python
from tensorflow.contrib import mpi

# Use `mpi.Session` instead of `tf.Session`
with mpi.Session() as session:
    rank = session.run(mpi.rank())
    print("My MPI Rank:", rank)

    if rank == 0:
        print("MPI Size:", session.run(mpi.size()))
```

@@rank
@@size

### Ring Allreduce and Allgather

When summing or averaging tensors across many processes, communication can
easily become a bottleneck. A naive implementation will send all the tensor
values to the same process, perform the reduction, and then broadcast the
values back to all other processes, effectively creating a synchronous
parameter server in one process. However, the process responsible for
performing the reduction will have to receive and send a massive amount of data
which scales with the number of processes *and* the number of parameters in the
model.

Instead of centralizing the reduction and having one primary reducer, we can
implement a distributed allreduce or allgather. A bandwidth-optimal allreduce
will end up sending 2(N - 1) values for every value in the input tensor,
and can be implemented with a ring allreduce [1]. (Intuitively, a linear reduce
requires at least (N - 1) sends between the different nodes, and a broadcast of
the result also requires (N - 1) sends, for a total of 2 (N - 1); these two
steps cannot be combined in a clever way to reduce the number of required
sends.) This module implements bandwidth-optimal ring allreduce and ring
allgather operations using MPI; by choosing a hardware-appropriate MPI
implementation (such as OpenMPI with CUDA-IPC support), you can train large
models with synchronous gradient descent with minimal communication overhead.

In addition to the `allreduce` and `allgather` functions, a convenience
`DistributedOptimizer` wrapper is provided to simplify using these functions
for reducing model gradients.

Example:

```python
import tensorflow as tf
from tensorflow.contrib import mpi

# Construct a simple linear regression model to optimize
W = tf.get_variable("W", shape=[20, 1], dtype=tf.float32)
B = tf.get_variable("B", shape=[1, 1], dtype=tf.float32)
inputs = tf.placeholder("Inputs", shape=[None, 20])
outputs = tf.placeholder("Outputs", shape=[None, 1])
loss = tf.nn.l2_loss(tf.matmul(inputs, W) + B - outputs)

# Training using MPI allreduce with DistributedOptimizer
optimizer = mpi.DistributedOptimizer(tf.train.AdamOptimizer())
train = optimizer.minimize(loss)

# Average loss over all ranks, for printing.
# Do not pass this to an optimizer!
avg_loss = mpi.allreduce(loss)

# On different ranks, feed different input data.
with mpi.Session() as session:
    rank = session.run(mpi.rank())
    batch_inputs, batch_outputs = construct_batch_for_rank(rank)
    feed_dict = {inputs: batch_inputs, outputs: batch_outputs}
    _, l = session.run([train, avg_loss], feed_dict=feed_dict)
    print("Average Loss:", l)
```

[1] Patarasuk, Pitch and Yuan, Xin. "Bandwidth Optimal All-reduce Algorithms
for Clusters of Workstations".

@@Session
@@DistributedOptimizer
@@allreduce
@@allgather
"""

import tensorflow as tf

import threading
import importlib
import os


_provider = None


def get_provider():
    global _provider
    if _provider is None:
        set_provider('horovod' if os.getenv('OMPI_COMM_WORLD_SIZE') is not None else 'dummy')
    return _provider


def set_provider(provider, force=False):
    global _provider
    if _provider is not None and not force:
        raise RuntimeError("%r already set as provider" % _provider)
    _provider = importlib.import_module('lib.ops.mpi.%s_provider' % provider)


def is_master():
    """
    Helper function to identify master
    """
    mpi_rank = os.getenv('OMPI_COMM_WORLD_RANK')
    return mpi_rank is None or mpi_rank == '0'


def is_distributed():
    """
    Helper function to identify if we are in distributed mode
    """
    mpi_size = os.getenv('OMPI_COMM_WORLD_SIZE')
    return mpi_size is not None and int(mpi_size) > 1


class Session(tf.Session):
    """A class for running TensorFlow operations, with copies of the same graph
    running distributed across different MPI nodes.

    The primary difference between `tf.Session` and `tf.contrib.mpi.Session` is
    that the MPI `Session` ensures that the `Session` options are correct for
    use with `tf.contrib.mpi`, and initializes MPI immediately upon the start
    of the session.
    """

    def __init__(self, gpu_group=None, gpu_group_size=1, target='', graph=None, config=None):
        """Creates a new TensorFlow MPI session.

        Unlike a normal `tf.Session`, an MPI Session may only use a single GPU,
        which must be specified in advance before the session is initialized.
        In addition, it only uses a single graph evaluation thread, and
        initializes MPI immediately upon starting.

        If no `graph` argument is specified when constructing the session,
        the default graph will be launched in the session. If you are
        using more than one graph (created with `tf.Graph()` in the same
        process, you will have to use different sessions for each graph,
        but each graph can be used in multiple sessions. In this case, it
        is often clearer to pass the graph to be launched explicitly to
        the session constructor.

        Args:
        gpu: (Optional.) The GPU index to use, or None for CPU only MPI.
        graph: (Optional.) The `Graph` to be launched (described above).
        config: (Optional.) A `ConfigProto` protocol buffer with configuration
        options for the session.
        """
        if config is None:
            config = tf.ConfigProto()

        if gpu_group is not None:
            config.gpu_options.visible_device_list = ','.join(str(gpu_group*gpu_group_size + d) for d in range(gpu_group_size))

        super(Session, self).__init__(target, graph, config=config)

        # Initialize MPI on the relevant device.
        with self.as_default():
            self.run(init())

        # Setup finalize status and lock to prevent double finalize call
        self._mpi_finalized = False
        self._mpi_finalize_lock = threading.Lock()

    def close(self):
        with self._mpi_finalize_lock:
            if not self._mpi_finalized:
                # Finalize MPI on the relevant device
                self.run(finalize())
                self._mpi_finalized = True

        super(Session, self).close()


###############################################################################
#
#  TensorFlow MPI operations
#
###############################################################################


def size(name=None):
    """An op which returns the number of MPI processes.

    This is equivalent to running `MPI_Comm_size(MPI_COMM_WORLD, ...)` to get the
    size of the global communicator.

    Returns:
    An integer scalar containing the number of MPI processes.
    """
    return get_provider().size(name)


def rank(name=None):
    """An op which returns the MPI rank of the calling process.

    This is equivalent to running `MPI_Comm_rank(MPI_COMM_WORLD, ...)` to get the
    rank of the current process in the global communicator.

    Returns:
    An integer scalar with the MPI rank of the calling process.
    """
    return get_provider().rank(name)


def local_rank(name=None):
    """An op which returns the local MPI rank of the calling process, within the
    node that it is running on. For example, if there are seven processes running
    on a node, their local ranks will be zero through six, inclusive.

    This is equivalent to running `MPI_Comm_rank(...)` on a new communicator
    which only includes processes on the same node.

    Returns:
    An integer scalar with the local MPI rank of the calling process.
    """
    return get_provider().local_rank(name=name)


def init(name=None):
    """An op which initializes MPI on the device on which it is run.

    All future MPI ops must be run on the same device that the `init` op was run
    on.
    """
    return get_provider().init(name)


def finalize(name=None):
    """An op which finalizes MPI on the device on which it is run.

    No future MPI ops must be run on the same device that the `finalize` op was run
    on.
    """
    return get_provider().finalize(name=name)


def allreduce(tensor, average=True, name=None):
    """Perform an MPI allreduce on a tf.Tensor or tf.IndexedSlices.

    Arguments:
        tensor: tf.Tensor, tf.Variable, or tf.IndexedSlices to reduce.
        The shape of the input must be identical across all ranks.
        average: If True, computes the average over all ranks.
                 Otherwise, computes the sum over all ranks.

    This function performs a bandwidth-optimal ring allreduce on the input
    tensor. If the input is an tf.IndexedSlices, the function instead does an
    allgather on the values and the indices, effectively doing an allreduce on
    the represented tensor.
    """
    return get_provider().allreduce(tensor, average, name)


def allgather(tensor, name=None):
    """An op which concatenates the input tensor with the same input tensor on
    all other MPI processes.

    The concatenation is done on the first dimension, so the input tensors on the
    different processes must have the same rank and shape, except for the first
    dimension, which is allowed to be different.

    Returns:
    A tensor of the same type as `tensor`, concatenated on dimension zero
    across all processes. The shape is identical to the input shape, except for
    the first dimension, which may be greater and is the sum of all first
    dimensions of the tensors in different MPI processes.
    """
    return get_provider().allgather(tensor, name)


def broadcast(tensor, name=None):
    """Broadcasts value of given tensor from coordinator node to all the others.

    Returns:
    Result of broadcast, same shape as `tensor`
    """
    return get_provider().broadcast(tensor, name=name)


def broadcast_var(ref, allow_uninitialized=False, name=None):
    """Broadcasts value of given variable from coordinator node to all the others.

    Returns:
    A mutable `tensor`, same as `ref`
    """
    return get_provider().broadcast_var(ref, allow_uninitialized=allow_uninitialized, name=name)


###############################################################################
#
#  Specific MPI operations on Python objects
#
###############################################################################


def broadcast_obj(obj, name=None):
    """
    Returns:
        Broadcasted object, same as input
    """
    return get_provider().broadcast_obj(obj, name)


def gather_obj(obj, name=None):
    """Gathers given Python object from all workers on the coordinator

    Returns:
      Gathered object on the coordinator (on all other workers None)
    """
    return get_provider().gather_obj(obj, name)


def scatter_obj(obj_array, name=None):
    """Scatters given array of Python objects to all workers from the coordinator

     Returns:
      Object on each worker
    """
    return get_provider().scatter_obj(obj_array, name)


def allgather_obj(obj, name=None):
    """Performs ALLGATHER on the given Python object

    Returns:
      Gathered object on all workers
    """
    return get_provider().allgather_obj(obj, name)
