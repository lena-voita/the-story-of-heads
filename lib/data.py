import numpy as np
import tensorflow as tf

import bintrees
import os
import sys
import random
import threading
import itertools
from .util import nested_pack, nested_flatten
from .ops import mpi


class TfUploader:

    def __init__(self, iterator, capacity, dtypes=None, shapes=None, session=None):
        self.session = session if session is not None else tf.get_default_session()
        self.empty = False

        # Detect dtypes from first iterator element
        if dtypes is None or shapes is None:
            # We need to wrap iterator access in session because it may call TF operations
            with self.session.as_default():
                try:
                    first = next(iterator)
                except StopIteration:
                    self.empty = True
                    return

            self.structure = first
            self.dtypes = tuple(e.dtype for e in nested_flatten(first))
            self.shapes = tuple(tuple(map(lambda x: None, e.shape)) for e in nested_flatten(first))
            self.iterator = itertools.chain([first], iterator)
        else:
            self.structure = dtypes
            self.dtypes = tuple(nested_flatten(dtypes))
            self.shapes = tuple(nested_flatten(shapes))
            self.iterator = iterator

        self.session_close_lock = threading.Lock()
        self.session_closed = False

        with tf.name_scope("uploader"):
            self.queue = tf.FIFOQueue(dtypes=self.dtypes, capacity=capacity)

            self.enqueue_inputs = [tf.placeholder(dtype=dt) for dt in self.dtypes]
            self.enqueue_op = self.queue.enqueue(self.enqueue_inputs)
            self.close_op = self.queue.close()

    def __enter__(self):
        if not self.empty:
            self.thread = threading.Thread(target=self._thread_main)
            self.thread.daemon = True
            self.thread.start()
        return self

    def __exit__(self, *args):
        if not self.empty:
            with self.session_close_lock:
                if not self.session_closed:
                    self.session.run(self.queue.close(True))
                    self.session_closed = True
            self.thread.join(1)
        return False

    def get_next(self):
        if self.empty:
            raise tf.errors.OutOfRangeError(None, None, "Queue is empty")
        res = self.queue.dequeue()
        if isinstance(res, list):
            for t, sh in zip(res, self.shapes):
                t.set_shape(sh)
            res = nested_pack(res, self.structure)
        return res

    def _thread_main(self):
        try:
            # We need to wrap iterator access in session because it may call TF operations
            with self.session.graph.as_default(), self.session.as_default():
                for t in self.iterator:
                    self.session.run(self.enqueue_op, feed_dict=dict(zip(self.enqueue_inputs, tuple(nested_flatten(t)))))

            with self.session_close_lock:
                self.session.run(self.close_op)
                self.session_closed = True
        except tf.errors.CancelledError:
            pass


class LastElement(object):
    """
    Class wrapping last element in RoundRobinIterator
    """
    def __init__(self, element=None):
        self.element = element

class RoundRobinIterator(object):
    """
    Class implementing Round-Robin iterator between coordinator and workers
    """
    def __init__(self, iterator=None, is_train=True, with_cost=False):
        self.iterator = iterator
        self.is_train = is_train
        self.with_cost = with_cost
        self.mpi_rank = os.getenv('OMPI_COMM_WORLD_RANK') or '0'
        self.mpi_size = os.getenv('OMPI_COMM_WORLD_SIZE') or '1'
        self.finish = False

    def __iter__(self):
        self.finish = False
        return self

    def __next__(self):
        if self.finish:
            # we should quit iterator
            raise StopIteration

        buf = None
        if self.mpi_rank == '0' or self.mpi_rank is None:
            # fill buffer with elements to scatter
            try:
                buf = []
                for _ in range(int(self.mpi_size)):
                    batch = next(self.iterator)
                    if self.with_cost:
                        if len(buf) == 0:  # On first element save coordinator cost
                            coord_cost = batch[-1]
                        batch = batch + (coord_cost,)
                    buf.append(batch)
            except StopIteration:
                # if iterator is out, scatter None values (during training) and
                # add None to missing workers
                if self.is_train:
                    buf = [LastElement()] * int(self.mpi_size)
                else:
                    for i in range(len(buf)):
                        buf[i] = LastElement(buf[i])
                    buf += [LastElement()] * (int(self.mpi_size) - len(buf))

        # scatter objects between workers
        value = mpi.scatter_obj(buf)
        if isinstance(value, LastElement):
            if value.element is None:
                raise StopIteration
            # remember to quit iterator at the next step
            self.finish = True
            value = value.element
        return value


class CostBufferIterator(object):
    """
    Class implementing CostBuffer iterator for fast finding of the batch with
    desired cost (useful for balancing batches)

    We assume inputs from the iterator passed in the constructor in the form:
    <batch> <cost> <coordinator cost>
    """
    def __init__(self, iterator=None, buf_size=1000):
        self.iterator = iterator
        self.buf_size = buf_size
        self.tree = bintrees.FastRBTree()
        self.coord_costs = []
        self.rng = random.Random(42)

    def __iter__(self):
        self.tree = bintrees.FastRBTree()
        self.coord_costs = []
        self.rng = random.Random(42)
        return self

    def __next__(self):
        # Warming up
        while len(self.coord_costs) < self.buf_size:
            try:
                batch, cost, coord_cost = next(self.iterator)
            except StopIteration:
                break
            if cost in self.tree:
                self.tree[cost].append(batch)
            else:
                self.tree[cost] = [batch]
            self.coord_costs.append(coord_cost)

        # No elements left - finish iteration
        if len(self.coord_costs) == 0:
            raise StopIteration

        # generate cost to choose and choose relevant batch
        index = self.rng.randrange(len(self.coord_costs))
        best_cost = self._find_best_match(self.coord_costs[index])
        batch = self.tree[best_cost][0]

        # remove selected items from structures
        del self.coord_costs[index]
        del self.tree[best_cost][0]
        if len(self.tree[best_cost]) == 0:
            del self.tree[best_cost]

        return batch

    def _find_best_match(self, cost):
        min_cost = self.tree.min_key()
        max_cost = self.tree.max_key()
        if cost <= min_cost:
            return min_cost
        if cost >= max_cost:
            return max_cost
        floor_cost = self.tree.floor_key(cost)
        ceil_cost = self.tree.ceiling_key(cost)
        return floor_cost if abs(ceil_cost - cost) < abs(floor_cost - cost) else ceil_cost


class ShuffleIterator(object):
    """
    Class implementing shuffling iterator via auxiliary buffer
    """
    def __init__(self, iterator, buf_size=1000):
        self.iterator = iterator
        self.buf_size = buf_size
        self.buf = []
        self.rng = random.Random(42)

    def __iter__(self):
        self.buf = []
        self.rng = random.Random(42)
        return self

    def __next__(self):
        # Return element from the previously shuffled buffer
        if len(self.buf) > 0:
            value = self.buf.pop()
            return value

        # Keep elements in the buffer
        while len(self.buf) < self.buf_size:
            try:
                value = next(self.iterator)
            except StopIteration:
                break
            self.buf.append(value)

        # No elements left - finish iteration
        if len(self.buf) == 0:
            raise StopIteration

        # Shuffle and return element from the buffer
        self.rng.shuffle(self.buf)
        value = self.buf.pop()
        return value


def pad_seq_list(array, sentinel):
    """
    Add padding, compose lengths
    """
    # Compute max length.
    maxlen = 0
    for seq in array:
        maxlen = max(maxlen, len(seq))

    # Pad.
    padded = []
    lens = []
    for seq in array:
        padding = maxlen - len(seq)
        padded.append(seq + [sentinel] * padding)
        lens.append(len(seq))

    return padded, lens
