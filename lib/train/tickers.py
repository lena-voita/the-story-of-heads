# Tickers
import sys
from functools import partial

import tensorflow as tf
import numpy as np
import time
import os
import re

from lib.train.saveload import get_model_variables, get_state_variables
from . import saveload
import lib
from ..util import *
from ..ops import mpi
from ..tools.average_npz import average_npzs

## ============================================================================
#                                   Ticker


class Ticker:
    """
    An interface for receiving notifications from optimizer. Several kinds of
    notifications are supported:

        - Training is started
        - Training has just processed a trainset batch
        - Devset loss is computed (happens every K training batches)
        - Training is complete

    User-defined tickers can be used e.g. for decaying learning rate in a
    custom manner. Also, see the collection of standard tickers.
    """

    init_priority = 0
    priority = 0

    def on_started(self, context):
        """
        Called when training starts.

        'context': TrainContext
        """
        pass

    def prepare_ingraph_ops(self):
        """
        Initializes ingraph operations of the ticker

        """
        self._ingraph_ops = self.on_train_batch_ingraph()

    def on_train_batch_ingraph(self):
        """
        Return a tensorflow operation that should be executed each time
        optimizer processes a training batch.
        """
        return []

    def before_train_batch(self):
        """
        Called each time before optimizer processing a training batch

        Return ingraph ops to run on this tick
        """
        return self._ingraph_ops

    def after_train_batch(self, ingraph_result):
        """
        Called each time optimizer processes a training batch.
        """
        pass

    def on_finished(self):
        """
        Called when the training is finished.
        """
        pass


class DistributedTicker(Ticker):
    """
    Has the same interface as Ticker, but on_* functions are called
    everywhere, not just on master.
    """

    pass


## ============================================================================
#                              Services & events


class Service:
    pass


class Subscriber:
    pass


class ModelService(Service):

    def get_problem(self):
        raise NotImplementedError()

    def get_model(self, name):
        raise NotImplementedError()


class TrainService(Service):

    def get_train_counters_ingraph(self):
        raise NotImplementedError()

    def get_train_loss_ingraph(self):
        raise NotImplementedError()

    def get_worker_batch_computation_time_ingraph(self):
        raise NotImplementedError()

    def get_worker_batch_time_ingraph(self):
        raise NotImplementedError()

    def get_worker_batch_lag_ingraph(self):
        raise NotImplementedError()


class RollbackService(Service):

    def rollback_to_or_before(self, step):
        raise NotImplementedError()


class SummaryWriterService(Service):

    def get_summary_writer(self):
        raise NotImplementedError()


class GlobalStepService(Service):

    def get_batch_no(self):
        raise NotImplementedError()

    def get_batch_no_ingraph(self):
        raise NotImplementedError()

    def get_global_step(self):
        raise NotImplementedError()

    def get_global_step_ingraph(self):
        raise NotImplementedError()

    def set_global_step(self, global_step):
        raise NotImplementedError()


class LearningRateService(Service):
    def __init__(self, learning_rate_fn):  # scale, learning_rate_policy_fn=None):
        self.multiplier = None
        self.learning_rate_fn = learning_rate_fn
        self.learning_rate = None

    # manage learning_rate
    def set_learning_rate(self, value):
        self.learning_rate = value

    def get_learning_rate(self):
        return self.learning_rate

    def get_learning_rate_ingraph(self):
        return self.learning_rate_fn()

    # Manage learning_rate multiplier
    def set_learning_rate_multiplier(self, new_multiplier):
        self.learning_rate *= new_multiplier / self.multiplier
        self.multiplier = tf.get_default_session().run(
            tf.assign(self.learning_rate_fn.multiplier_var, new_multiplier)
        )

    def get_learning_rate_multiplier(self):
        if self.multiplier is None:
            self.multiplier = tf.get_default_session().run(self.learning_rate_fn.multiplier_var)
        return self.multiplier


class DevSubscriber(Subscriber):

    def prepare_dev_graph(self, dev_name, dev_counters, dev_loss):
        """
        Called at the start of process to prepare part of tensorflow graph
        for processing dev results
        """
        pass

    def before_dev_run(self, dev_name):
        """
        Called each dev step to get lst of ops which should be evaluated this time
        Call result may differ between steps (for example, if you want to do something only once in several steps)
        This call MUST NOT create new graph operations, only return those already prepared
        """
        return []

    def after_dev_run(self, dev_name, dev_run_values):
        """
        Called each dev step after evaluating ops and provides their results
        """
        pass


## ============================================================================
#                             Standard tickers

# -
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


class Summary(DistributedTicker, SummaryWriterService, DevSubscriber):
    """
    Compute and store train batch summary every `every_steps` training batches.
    Compute and store dev summary every time dev is computed.
    Compute and store parameter summary once in a while.
    """

    def __init__(self, folder, every_steps=1, params_every_steps=None, params_every_minutes=None):
        self.folder = folder
        self.params_every_steps = params_every_steps
        self.params_every_minutes = params_every_minutes
        self.every_steps = every_steps
        self.dev_summary_graphs = {}

    def on_started(self, context):
        if mpi.is_master() and not os.path.exists(self.folder):
            os.makedirs(self.folder)

        self.context = context
        self.writer = tf.summary.FileWriter(self.folder)
        self.problem = context.get_problem()

        if mpi.is_master():
            self.writer.add_graph(tf.get_default_graph())

        self.is_it_time_yet_for_params = _IsItTimeYet(
            context, self.params_every_steps, self.params_every_minutes)
        self.is_it_time_yet_for_summary = _IsItTimeYet(
            context, self.every_steps, None)  # None for minutes as we don't use them for training summary

    def on_finished(self):
        self.writer.close()

    def prepare_ingraph_ops(self):
        counters = self.context.get_train_counters_ingraph()
        learning_rate = self.context.get_learning_rate_ingraph()

        self._summary_ops = merge_summaries(
            [tf.summary.scalar('Train/LearningRate', learning_rate)] +
            self.problem.summary_multibatch(counters, 'Train', is_train=True) +
            tf.get_collection(lib.meta.SUMMARIES_ZOO)
        )

        self._params_summary_ops = merge_summaries(
            [tf.summary.histogram(v.name, v) for v in tf.model_variables()] +
            self.problem.params_summary() +
            tf.get_collection(lib.meta.PARAMS_SUMMARIES)
        )

    def before_train_batch(self):
        self.start = time.time()

        # return dummy if it is not time to compute summary yet
        ops = {}
        if self.is_it_time_yet_for_summary(True):
            ops['summary'] = self._summary_ops
        if self.is_it_time_yet_for_params():
            ops['params_summary'] = self._params_summary_ops
        return ops

    def prepare_dev_graph(self, dev_name, dev_counters, dev_loss):
        self.dev_summary_graphs[dev_name] = merge_summaries(self.problem.summary_multibatch(dev_counters, dev_name,
                                                                                            is_train=False))

    def before_dev_run(self, dev_name):
        return self.dev_summary_graphs[dev_name]

    def after_dev_run(self, dev_name, dev_run_value):
        if not lib.ops.mpi.is_master():
            return
        self.writer.add_summary(dev_run_value, self.context.get_global_step())

    def after_train_batch(self, ingraph_result):
        if not lib.ops.mpi.is_master():
            return

        step = self.context.get_global_step()

        if 'summary' in ingraph_result:
            self.writer.add_summary(ingraph_result['summary'], step)

        if 'params_summary' in ingraph_result:
            self.writer.add_summary(ingraph_result['params_summary'], step)

    def get_summary_writer(self):
        return self.writer

    def _flush(self):
        self.writer.flush()


# - Save  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


class SaveLoad(DistributedTicker, RollbackService):

    init_priority = -100
    priority = 90

    def __init__(self, folder, every_steps=None, every_minutes=None, skip_train_data=True,
                 keep_checkpoints_max=None, pre_init_model_checkpoint=None, pre_init_state_checkpoint=None,
                 avg_checkpoint_every=None, avg_last_checkpoints=None):
        self.folder = folder
        self.every_steps = every_steps
        self.every_minutes = every_minutes
        self.skip_train_data = skip_train_data
        self.keep_checkpoints_max = keep_checkpoints_max
        self.pre_init_model_checkpoint = pre_init_model_checkpoint
        self.pre_init_state_checkpoint = pre_init_state_checkpoint
        self.avg_checkpoint_every = avg_checkpoint_every
        self.avg_last_checkpoints = avg_last_checkpoints
        assert (avg_last_checkpoints is None) == (avg_checkpoint_every is None)
        if avg_checkpoint_every is not None:
            assert avg_checkpoint_every >= avg_last_checkpoints

    def average_last_checkpoints(self):
        if not lib.ops.mpi.is_master():
            return
        checkpoint_names = []
        for label in self._get_sorted_labels()[::-1][:self.avg_last_checkpoints]:
            checkpoint_names.append(os.path.join(self.folder, 'model-%s.npz' % label))
        averaged_checkpoint = average_npzs(checkpoint_names)
        print("\n\t lib.train.tickers.Saveload.average_last_checkpoints")
        print('\n'.join(checkpoint_names))
        return averaged_checkpoint

    def on_started(self, context):
        if lib.ops.mpi.is_master() and not os.path.exists(self.folder):
            os.makedirs(self.folder)

        # 1. Init.
        self.context = context

        # 2. We can't set these functions here yet because global step may change
        self.is_it_time_yet = None

        # 3. Load checkpoint and sync global state
        self._load_or_save_initial_checkpoints()
        self._broadcast_global_variables()
        self.context.set_global_step(tf.get_default_session().run(tf.get_collection(tf.GraphKeys.GLOBAL_STEP)[0]))

        # 4. Skip training data if necessary
        batch_no = tf.get_default_session().run(self.context.get_batch_no_ingraph())
        if self.skip_train_data and batch_no > 0:
            self.context.skip_train_data(batch_no)

        # 5. Set is_it_time_yet functions with correct global step
        self.is_it_time_yet = _IsItTimeYet(
            self.context, self.every_steps, self.every_minutes)

    def after_train_batch(self, ingraph_result):
        if self.is_it_time_yet():
            self._save(self.context.get_global_step())
            if (self.avg_last_checkpoints is not None) and\
                (int(int(self._find_latest_label()) / self.every_steps) % self.avg_last_checkpoints == 0):
                print("\n AVERAGING CHECKPOINTS: \n")
                averaged_checkpoint = self.average_last_checkpoints()
                latest_label = self._find_latest_label()
                print("\n\t latest_label: {}".format(latest_label))
                if latest_label is None:
                    return
                model_path = os.path.join(self.folder, 'model-%s.npz' % latest_label)
                np.savez(model_path, **averaged_checkpoint)
                self._load(latest_label)

    def on_finished(self):
        self._save('final')

    def rollback_to_or_before(self, max_step):
        global_step = self.context.get_global_step()

        label = self._find_latest_label(max_step) if lib.ops.mpi.is_master() else ''

        if label is None:
            raise RuntimeError("No checkpoint to rollback found!")

        self._load(label)
        self._broadcast_global_variables()
        self.context.set_global_step(global_step)

    def _load_or_save_initial_checkpoints(self):
        stop_decision = None

        if lib.ops.mpi.is_master():
            # Find latest model.
            #     - 'model-init.npz' < 'model-N.npz' < 'model-final.npz'
            #     - 'model-latest.npz' is ignored.
            latest_label = self._find_latest_label()
            if latest_label is not None:
                # Load parameters if we found a checkpoint.
                self._load(latest_label)

                # End training if loaded final checkpoint.
                if latest_label == 'final':
                    model_path = os.path.join(self.folder, 'model-%s.npz' % latest_label)
                    state_path = os.path.join(self.folder, 'state-%s.npz' % latest_label)
                    stop_decision = 'Found final checkpoints %r and %r' % (model_path, state_path)
            else:
                # If didn't find a checkpoint, load pre-init and save initial parameters
                self._load_pre_init_checkpoints()
                self._save('init')
                self._save_graph()

        # Distribute stop decision to other workers
        stop_decision = lib.ops.mpi.broadcast_obj(stop_decision, name="stop_decision")
        if stop_decision is not None:
            self.context.stop_training(stop_decision)

    def _find_latest_label(self, max_label=None):
        # Get all labels sorted in ascending order and retrieve last label no more than max_label
        for label in self._get_sorted_labels()[::-1]:
            if max_label is None or not self._label_gt(label, max_label):
                return label
        return None

    def _get_sorted_labels(self):
        """ Get list of sorted labels

        'init' is prepended (if exists) and 'final' is appended (if exists)
        """
        init_exist = final_exist = False
        labels = []
        for fname in os.listdir(self.folder):
            # Skip invalid filenames.
            if not fname.startswith('model-') or not fname.endswith('.npz'):
                continue
            label = fname[len('model-'):-len('.npz')]
            if label not in ['init', 'final'] and not label.isdigit():
                continue

            # Skip models without state.
            model_path = os.path.join(self.folder, fname)
            state_path = os.path.join(self.folder, 'state-%s.npz' % label)
            if not os.path.isfile(model_path) or not os.path.isfile(state_path):
                continue

            if label == 'init':
                init_exist = True
            elif label == 'final':
                final_exist = True
            else:
                labels.append(label)

        return (['init'] if init_exist else []) + sorted(labels, key=lambda x: int(x)) + (['final'] if final_exist else [])

    def _label_gt(self, l1, l2):
        """ Check if first label greater then the second """

        # If labels are equal first is not greater
        # If first label is 'init' it can't be greater then any other
        # If second is 'final' it can't be less
        if l1 == l2 or l1 == 'init' or l2 == 'final':
            return False

        if l1 == 'final' or l2 == 'init':
            return True

        # Now we are sure what both labels are numbers
        return int(l1) > int(l2)

    def _save_graph(self):
        if lib.ops.mpi.is_master():
            print('! Saving graph to %r' % os.path.join(self.folder, 'graph.pbtxt'), file=sys.stderr, flush=True)
            tf.train.write_graph(tf.get_default_session().graph_def, self.folder, 'graph.pbtxt')

    def _save(self, label):
        # Save (only on master)
        if lib.ops.mpi.is_master():
            model_vars = saveload.get_model_variables()
            state_vars = saveload.get_state_variables()
            model_path = os.path.join(self.folder, 'model-%s.npz' % label)
            state_path = os.path.join(self.folder, 'state-%s.npz' % label)
            print('! Saving model to %r' % model_path, file=sys.stderr, flush=True)
            print('! Saving state to %r' % state_path, file=sys.stderr, flush=True)
            saveload.save(model_path, model_vars)
            saveload.save(state_path, state_vars)

            if self.keep_checkpoints_max:
                self._remove_old_labels()

            # Create symlink 'model-latest.npz'.
            sym_from = 'model-%s.npz' % label
            sym_to = os.path.join(self.folder, 'model-latest.npz')
            try:
                os.unlink(sym_to)
            except OSError as e:
                if e.errno != 2: # File not found.
                    raise
            os.symlink(sym_from, sym_to)

    def _remove_old_labels(self):
        """Removes oldest labels"""
        labels = self._get_sorted_labels()
        for index in range(0, len(labels) - self.keep_checkpoints_max):
            os.remove(os.path.join(self.folder, 'model-%s.npz' % labels[index]))
            os.remove(os.path.join(self.folder, 'state-%s.npz' % labels[index]))

    def _load(self, label):
        # Load (only from master)
        if lib.ops.mpi.is_master():
            model_vars = saveload.get_model_variables()
            state_vars = saveload.get_state_variables()
            model_path = os.path.join(self.folder, 'model-%s.npz' % label)
            state_path = os.path.join(self.folder, 'state-%s.npz' % label)
            print('! Loading model from %r' % model_path,
                file=sys.stderr, flush=True)
            saveload.load(model_path, model_vars)
            print('! Loading state from %r' % state_path,
                file=sys.stderr, flush=True)
            saveload.load(state_path, state_vars)
            print('! Loading - DONE', file=sys.stderr, flush=True)

            uninitialized_names = sorted(tf.get_default_session().run(tf.report_uninitialized_variables()))
            if len(uninitialized_names) > 0:
                print('! Uninitialized variables after loading checkpoint: %s' % str(uninitialized_names), file=sys.stderr, flush=True)

    def _load_pre_init_checkpoints(self):
        if self.pre_init_model_checkpoint:
            print('! Loading pre-init model from %r' % self.pre_init_model_checkpoint,
                file=sys.stderr, flush=True)
            saveload.load(self.pre_init_model_checkpoint, saveload.get_model_variables())

        if self.pre_init_state_checkpoint:
            print('! Loading pre-init state from %r' % self.pre_init_state_checkpoint,
                file=sys.stderr, flush=True)
            saveload.load(self.pre_init_state_checkpoint, saveload.get_state_variables())

    def _broadcast_global_variables(self):
        ops = []
        for var in tf.global_variables():
            with tf.device(var.device):
                ops.append(lib.ops.mpi.broadcast_var(var, name=re.sub('\\W', '_', var.name)))
        tf.group(*ops).run()


# - DevLoss - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


class DevLossTicker(DistributedTicker):
    """
    - Compute devset loss once in a while.
    - Print devset loss to stderr every time train() computes it.
    - Print a dot  .  every trainset batch.
    """

    def __init__(self, devset, name='Dev', every_steps=None, every_minutes=None, initial=False):
        self.devset = devset
        self.name = name
        self.every_steps = every_steps
        self.every_minutes = every_minutes
        self.initial = initial

    def on_started(self, context):
        self.problem = context.get_problem()
        self.devset_batches = list(self.devset)
        self.subscribers = context.get_subscribers(DevSubscriber)

        if len(self.devset_batches) > 0:
            with tf.name_scope("%s/loss" % self.name), lib.meta.lock_collections([lib.meta.SUMMARIES_ZOO,
                                                                                  lib.meta.PARAMS_SUMMARIES]):

                self.dev_batch_inp = nested_map(lambda e: tf.placeholder(dtype=tf.as_dtype(e.dtype), shape=[None] * len(e.shape)), self.devset_batches[0])

                dev_batch_copy = nested_map(lambda x: x, self.dev_batch_inp)
                self.dev_batch_counters_op = self.problem.batch_counters(dev_batch_copy, is_train=False)
                # note: we send copy of batch_counters dict to avoid bugs if user changes it inside batch_counters

        self.subscriber_ops = None  # Will be initialized later
        self.dev_loss_op = None

        self.is_it_time_yet = _IsItTimeYet(
            context, self.every_steps, self.every_minutes)

        # Score devset after initialization if option passed (and we are not loading some non-init checkpoint)
        if self.initial and context.get_global_step() == 0:
            self._score()

    def after_train_batch(self, ingraph_result):
        if self.is_it_time_yet():
            self._score()

    def _score(self):
        # Iterate over dev set and compute counters for each batch
        counters = []
        for batch in self.devset_batches:
            feed_dict = dict(zip(nested_flatten(self.dev_batch_inp), nested_flatten(batch)))
            counters.append(tf.get_default_session().run(self.dev_batch_counters_op, feed_dict=feed_dict))

        # Allgather counters
        counters = lib.ops.mpi.allgather_obj(counters)
        counters = [x for c in counters for x in c]

        # Stack counters from different batches
        counters = nested_map(lambda *x: np.stack(x), *counters)

        # Delayed initialization of subscriber ops to ensure what subscribers are initialized
        if self.dev_loss_op is None:
            with tf.name_scope("%s/loss" % self.name):
                self.dev_counters_inp = nested_map(lambda e: tf.placeholder(dtype=tf.as_dtype(e.dtype),
                                                                            shape=[None] * len(e.shape)), counters)
                self.dev_loss_op = self.problem.loss_multibatch(self.dev_counters_inp, is_train=False)

            # Prepare graphs in all subscribers
            for subscriber in self.subscribers:
                subscriber.prepare_dev_graph(self.name, self.dev_counters_inp, self.dev_loss_op)

        # Prepare list of ops
        ops = []
        for subscriber in self.subscribers:
            ops.append(subscriber.before_dev_run(self.name))
        ops.append(self.dev_loss_op)

        # Compute devloss and subscriber ops
        feed_dict = dict(zip(nested_flatten(self.dev_counters_inp), nested_flatten(counters)))
        results = tf.get_default_session().run(ops, feed_dict=feed_dict)
        dev_loss = results.pop()

        if lib.ops.mpi.is_master():
            # Print dev loss only on the master
            print('%f' % dev_loss, file=sys.stderr, flush=True)
        for subscriber, result in zip(self.subscribers, results):
            subscriber.after_dev_run(self.name, result)


# - DecayLearningRate - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


class DecayLearningRate(DistributedTicker, DevSubscriber):
    """
    Decay learning rate and rollback.
    """

    priority = 80

    def __init__(self, after_steps, rollback=True, decay_by=2**0.5, dev_name='Dev'):
        self.after_steps = after_steps
        self.decay_by = decay_by
        self.rollback = rollback

        # Track min loss only on the master
        self.min_dev_loss = None
        self.min_dev_loss_var = tf.get_variable(
            'min_dev_loss', [], tf.float32,
            initializer=tf.constant_initializer(np.inf), trainable=False
            )

        self.min_dev_loss_global_step = None
        self.min_dev_loss_global_step_var = tf.get_variable(
            'min_dev_loss_global_step', [], tf.int64,
            initializer=tf.constant_initializer(0), trainable=False
            )

        self.min_dev_loss_inp = tf.placeholder(tf.float32, shape=[])
        self.min_dev_loss_global_step_inp = tf.placeholder(tf.int64, shape=[])
        self.update_vars = [
            tf.assign(self.min_dev_loss_var, self.min_dev_loss_inp),
            tf.assign(self.min_dev_loss_global_step_var, self.min_dev_loss_global_step_inp)
        ]

        self.dev_name = dev_name

    def on_started(self, context):
        self.context = context

    def prepare_dev_graph(self, dev_name, dev_counters, dev_loss):
        if dev_name != self.dev_name:
            return
        self.dev_loss_ingraph = dev_loss

    def before_dev_run(self, dev_name):
        if dev_name != self.dev_name:
            return []
        return self.dev_loss_ingraph

    def after_dev_run(self, dev_name, dev_loss):
        if dev_name != self.dev_name:
            return

        global_step = self.context.get_global_step()

        # Track min loss only on the master
        if self.min_dev_loss is None or dev_loss < self.min_dev_loss:
            self.min_dev_loss, self.min_dev_loss_global_step = \
                tf.get_default_session().run(self.update_vars, feed_dict={
                    self.min_dev_loss_inp: dev_loss,
                    self.min_dev_loss_global_step_inp: global_step})

        # Decay learning rate if necessary
        if global_step - self.min_dev_loss_global_step >= self.after_steps:
            new_learning_rate_multiplier = self.context.get_learning_rate_multiplier() / self.decay_by

            if lib.ops.mpi.is_master():
                print("! Decaying learning rate multiplier to %f" % new_learning_rate_multiplier, file=sys.stderr, flush=True)

            if self.rollback:
                self.context.rollback_to_or_before(self.min_dev_loss_global_step)
                self.min_dev_loss_global_step = global_step

            self.context.set_learning_rate_multiplier(new_learning_rate_multiplier)


# - LearningRateStopper - - - - - - - - - - - - - - - - - - - - - - - - - - - -


class LearningRateStopper(DistributedTicker):
    """
    Stop training when learning rate becomes too low.
    """

    priority = 100

    def __init__(self, threshold=None, times=None, min_steps=None):
        self.threshold = threshold
        self.times = times
        self.min_steps = min_steps

    def on_started(self, context):
        self.context = context
        try:
            self.initial_lr = self.context.get_learning_rate()
        except NotImplementedError:
            msg  = '! LearningRateStopper is useless: '
            msg += 'algorithm has no learning rate'
            print(msg, file=sys.stderr, flush=True)
            self.initial_lr = None

    def after_train_batch(self, ingraph_result):
        lr = self.context.get_learning_rate()

        if lr is None:
            raise Exception("LearningRateStopper:: lr expected to be not None")
        if self.initial_lr is None:
            self.initial_lr = lr
            return

        global_step = self.context.get_global_step()
        if self.min_steps is not None and global_step < self.min_steps:
            return

        if self.threshold is not None and lr <= self.threshold:
            msg = 'Learning rate %f below threshold %f'
            self.context.stop_training(msg % (lr, self.threshold))

        if self.times is not None and self.initial_lr * (1 + 1e-5) >= lr * self.times:
            msg = 'Learning rate %f is %f times less than initial %f'
            self.context.stop_training(msg % (lr, self.times, self.initial_lr))


# - GlobalStepStopper - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


class GlobalStepStopper(DistributedTicker):
    """
    Stop training after N batches.
    """

    priority = 100

    def __init__(self, num_steps):
        self.num_steps = num_steps

    def on_started(self, context):
        self.context = context

    def after_train_batch(self, _ingraph_result):
        step = self.context.get_global_step()
        if step >= self.num_steps:
            self.context.stop_training('Made %i steps' % step)


# - NanInfSpotter - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


class NanInfSpotter(DistributedTicker, DevSubscriber):
    """
    Spots Nan or Inf values in any tensor during training.
    """
    def __init__(self, debug=False, op_names=()):
        """
        debug: turns on slow debug mode
        op_names: names of operations to monitor for nan/inf values
        """
        self.debug = debug
        self.op_names = list(op_names)

        self.dev_graphs = {}

    def on_started(self, context):
        self.context = context

    def on_train_batch_ingraph(self):
        if self.debug:
            # Slow mode - every floating tensor is wrapped
            return self._add_check_numerics_ops(self.op_names)
        return self.context.get_train_loss_ingraph()

    def after_train_batch(self, ingraph_result):
        if np.isnan(ingraph_result) or np.isinf(ingraph_result):
            raise tf.errors.InvalidArgumentError(None, "Train Loss", "Train Loss: Nan or inf values spotted")

    def prepare_dev_graph(self, dev_name, dev_counters, dev_loss):
        if self.debug:
            # Slow mode - every floating tensor is wrapped
            self.dev_graphs[dev_name] = self._add_check_numerics_ops(self.op_names)
        else:
            self.dev_graphs[dev_name] = dev_loss

    def before_dev_run(self, dev_name):
        return self.dev_graphs[dev_name]

    def after_dev_run(self, dev_name, dev_run_values):
        if np.isnan(dev_run_values) or np.isinf(dev_run_values):
            raise tf.errors.InvalidArgumentError(None, "Dev Loss", "Dev Loss: Nan or inf values spotted")

    def _add_check_numerics_ops(self, op_names):
        check_op = []
        for op_name in op_names:
            for op in tf.get_default_graph().get_operation_by_name(op_name):
                for output in op.outputs:
                    if output.dtype in [tf.float16, tf.float32, tf.float64]:
                        message = op_name + ':' + str(output.value_index)
                        with tf.control_dependencies(check_op):
                            check_op = [tf.check_numerics(output, message=message)]
        return tf.group(*check_op)


# - DivergenceSpotter - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


class DivergenceSpotter(DistributedTicker):

    priority = 120

    def __init__(self, every_steps=None, every_minutes=None, sync=False, debug=False):
        self.every_steps = every_steps
        self.every_minutes = every_minutes
        self.sync = sync
        self.debug = debug

    def on_started(self, context):
        self.context = context
        self.is_it_time_yet = _IsItTimeYet(context, self.every_steps, self.every_minutes)

        self.var_list = get_model_variables() | get_state_variables()
        self.op_list = []

        for var in self.var_list:
            dtype = var.dtype.base_dtype

            if dtype == tf.float32:
                op = tf.bitcast(var, tf.int32)
            elif dtype == tf.float64:
                op = tf.bitcast(var, tf.int64)
            elif dtype in [tf.int8, tf.int16, tf.int32, tf.int64, tf.uint8, tf.uint16]:
                op = var
            else:
                raise RuntimeError("Unknown var dtype: %s" % var.op.dtype)

            self.op_list.append(lib.ops.mpi.allgather(tf.expand_dims(tf.reduce_sum(op), axis=0),
                                                      name=re.sub('\\W', '_', var.name) + '_sum_allgather'))

    def after_train_batch(self, ingraph_result):
        if not self.is_it_time_yet():
            return

        values = tf.get_default_session().run(self.op_list)
        diverged_vars = []

        for var, vals in zip(self.var_list, values):
            if any(v != vals[0] for v in vals):
                if self.debug and lib.ops.mpi.is_master():
                    print("Detected divergence in %s: %s" % (var.name, str(vals)), file=sys.stderr, flush=True)
                diverged_vars.append(var)

        if self.sync and len(diverged_vars) > 0:
            if self.debug and lib.ops.mpi.is_master():
                print("Synchronizing diverged variables: %s" % str([v.name for v in diverged_vars]),
                      file=sys.stderr, flush=True)

            ops = []
            for var in diverged_vars:
                ops.append(tf.assign(var, lib.ops.mpi.allreduce(var, average=True,
                                                                name=re.sub('\\W', '_', var.name) + '_sync')))
            tf.group(*ops).run()


class LearningRateFn:
    def __init__(self, policy, scale, **kwargs):
        bound_policy_fn = partial(learning_rate_policy_fn, policy=policy, scale=scale, **kwargs)
        self.policy_fn = lambda global_step_var: bound_policy_fn(global_step=global_step_var)

        self.multiplier_var = tf.get_variable(
            'learning_rate_multiplier', [], tf.float32,
            initializer=tf.constant_initializer(1.0),
            trainable=False)

        self.learning_rate_ingraph = None

    def __call__(self):
        # create policy here
        if self.learning_rate_ingraph is None:
            self.learning_rate_ingraph = self.multiplier_var * self.policy_fn(tf.train.get_global_step())
        return self.learning_rate_ingraph


def learning_rate_policy_fn(policy, scale, global_step, **kwargs):
    with tf.name_scope("learning_rate_fn"):
        if policy == 'constant':
            new_learning_rate = tf.constant(scale, dtype=tf.float32)
        elif policy == 'exponential':
            new_learning_rate = tf.train.exponential_decay(
                scale, global_step,
                kwargs['decay_steps'], kwargs['decay_rate'],
                staircase=kwargs.get('staircase', False))
        elif policy == 'inverse_time':
            new_learning_rate = tf.train.inverse_time_decay(
                scale, global_step,
                kwargs['decay_steps'], kwargs['decay_rate'],
                staircase=kwargs.get('staircase', False))
        elif policy == 'natural_exp':
            new_learning_rate = tf.train.natural_exp_decay(
                scale, global_step,
                kwargs['decay_steps'], kwargs['decay_rate'],
                staircase=kwargs.get('staircase', False))
        elif policy == 'polynomial':
            new_learning_rate = tf.train.polynomial_decay(
                scale, global_step,
                kwargs['decay_steps'], kwargs['end_learning_rate'],
                power=kwargs.get('power', 1.0), cycle=kwargs.get('cycle', False))
        elif policy == 'warmup_expup':
            new_learning_rate = scale * tf.where(
                global_step > kwargs['decay_steps'],
                tf.exp( tf.to_float(global_step) / kwargs['decay_steps'] - 1.0 ), # exp growth
                tf.to_float(global_step) / kwargs['decay_steps']  #linear growth
            )
        elif policy == 'warmup_const':
            new_learning_rate = scale * tf.where(
                global_step > kwargs['decay_steps'],
                tf.constant(1.0, dtype=tf.float32), # const
                tf.to_float(global_step) / kwargs['decay_steps']  #linear growth
            )
        elif policy == 'wait_const':
            new_learning_rate = scale * tf.where(
                global_step > kwargs['decay_steps'],
                tf.constant(1.0, dtype=tf.float32), # const
                tf.constant(0.0, dtype=tf.float32), # zero LR, - Wait for Adam stats
            )
        elif policy == 'warmup_inverse_sqrt_time':
            new_learning_rate = scale * \
                tf.minimum(
                    tf.to_float(global_step + 1) ** -0.5,
                    tf.to_float(global_step + 1) * kwargs['decay_steps'] ** -1.5) *\
                kwargs['decay_steps'] ** 0.5
        elif policy == 't2t_noam':
            new_learning_rate = scale * \
                tf.minimum(
                    tf.to_float(global_step + 1) ** -0.5,
                    tf.to_float(global_step + 1) * kwargs['decay_steps'] ** -1.5) * \
                kwargs['hid_size'] ** -0.5
        else:
            raise ValueError("Wrong policy for learning rate scheduling specified")

        return new_learning_rate

# - - - - - - - - - - - - - - - - - - - - - - - - - - - -


class TimeTicker(Ticker):
    """
    Prints elapsed time every N steps
    """

    priority = 999

    def __init__(self, every_steps=100):
        self.every_steps = every_steps

    def on_started(self, context):
        self.context = context

        self.is_it_time_yet = _IsItTimeYet(context, self.every_steps, None)

        self.last_time = time.time()
        self.last_step = context.get_global_step()

    def after_train_batch(self, ingraph_result):
        if not self.is_it_time_yet():
            return

        cur_time = time.time()
        cur_step = self.context.get_global_step()

        elapsed_time = cur_time - self.last_time
        elapsed_steps = cur_step - self.last_step

        print("\nStep %d: made %d steps in %.3f seconds (%.3f steps/sec)" % (cur_step, elapsed_steps, elapsed_time, elapsed_steps / elapsed_time), file=sys.stderr, flush=True)

        self.last_time = cur_time
        self.last_step = cur_step


## ============================================================================
#                                Utilities


class _IsItTimeYet:
    """
    Class that tells whether it's time to do some action.

    It tracks number of steps and number of minutes passed since the last
    time we executed the action.
    """

    def __init__(self, context, every_steps=None, every_minutes=None):
        self.context = context
        self.every_steps = every_steps
        self.every_minutes = every_minutes
        self.last_step = self.context.get_global_step()
        self.last_time = time.time()

    def __call__(self, true_on_same=False):
        """
        Tells whether it's time to do the action.

        It's assumed that the action is executed after is() call, so the
        tracker updates its timestamp and step number.
        """
        N = self.every_steps
        if N:
            if self.context.get_global_step() - self.last_step >= N:
                self._do()
                return True
            elif true_on_same and self.context.get_global_step() == self.last_step:
                # if we call this function on the same tick, it will return True
                return True

        if self.every_minutes:
            if (time.time() - self.last_time) / 60 > self.every_minutes:
                self._do()
                return True

        return False

    def _do(self):
        self.last_step = self.context.get_global_step()
        self.last_time = time.time()
