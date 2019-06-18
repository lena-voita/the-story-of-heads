# Training routine

from collections import defaultdict
import tensorflow as tf
import sys
import inspect

import lib
from lib.train.saveload import initialize_uninitialized_variables
from . import algorithms, saveload, tickers
from .problem import Problem, SimpleProblem
from .tickers import Ticker, DistributedTicker, TrainService, LearningRateService, ModelService, \
    LearningRateFn, SummaryWriterService, GlobalStepService

from ..session import profile_scope
from ..data import TfUploader
from ..util import nested_map

class DuplicateServiceError(Exception):
    pass

class DuplicateMethodError(Exception):
    pass


# Main train loop


def train(problem, algorithm, iterator, tickers, tick_every_steps=0):
    uploader = TfUploader(iterator, capacity=5)
    if uploader.empty:
        raise RuntimeError("Trainset is empty")

    global_step_ticker = _GlobalStepService(tick_every_steps)
    train_ticker = _TrainTicker(problem, algorithm, uploader)

    tickers = [global_step_ticker, train_ticker] + sorted(tickers, key=lambda t: t.priority)

    if not lib.ops.mpi.is_master():
        tickers = [t for t in tickers if isinstance(t, DistributedTicker)]

    real_tickers = [t for t in tickers if isinstance(t, Ticker)]

    session = tf.get_default_session()
    context = _TrainContext(uploader.iterator, tickers)

    initialize_uninitialized_variables()

    for ticker in sorted(real_tickers, key=lambda t: t.init_priority):
        ticker.on_started(context)
    # second loop because tickers can depend on each other
    for ticker in real_tickers:
        ticker.prepare_ingraph_ops()

    with uploader:
        try:
            while not context.should_stop:
                batch_evals = []
                for ticker in real_tickers:
                    batch_evals.append(ticker.before_train_batch())

                with profile_scope(level=1):
                    if tick_every_steps == 0:
                        batch_results = session.run(batch_evals, feed_dict={
                            global_step_ticker.global_step_ingraph: global_step_ticker.global_step,
                            global_step_ticker.batch_no_ingraph: global_step_ticker.batch_no
                        })
                    else:
                        batch_results = session.run(batch_evals, feed_dict={
                            global_step_ticker.global_step_ingraph: global_step_ticker.global_step,
                            global_step_ticker.batch_no_ingraph: global_step_ticker.batch_no,
                            global_step_ticker.tick_no_ingraph: global_step_ticker.tick_no
                        })

                for ticker, result in zip(real_tickers, batch_results):
                    ticker.after_train_batch(result)

        except tf.errors.OutOfRangeError:
            pass

    for ticker in real_tickers:
        ticker.on_finished()


## ============================================================================
#                              Internal functions


def _get_classes(cls, desired_cls):
    bases = cls.__bases__

    if desired_cls in bases:
        yield cls
    else:
        for base in bases:
            yield from _get_classes(base, desired_cls)


class _TrainContext(object):

    def __init__(self, iterator, objects):
        self._register_providers(objects)
        self.subscribers = self._register_subscribers(objects)
        self.iterator = iterator
        self.should_stop = False

    def stop_training(self, reason):
        print("Stopping because of %s" % reason, file=sys.stderr)
        self.should_stop = True

    def skip_train_data(self, batches):
        print("! Skipping %d batches..." % batches, file=sys.stderr, flush=True, end='')
        for i in range(batches):
            if i > 1000 and (i & (i - 1)) == 0:
                print(" %d" % i, file=sys.stderr, flush=True, end='')
            next(self.iterator)
        print(" done", file=sys.stderr, flush=True)

    def _register_providers(self, objects):
        service_providers = {}
        method_services = {}

        for obj in objects:
            for srv_class in _get_classes(type(obj), tickers.Service):
                if srv_class in service_providers:
                    raise DuplicateServiceError("Multiple providers for service %s detected: %s and %s" % (srv_class, service_providers[srv_class], obj))

                service_providers[srv_class] = obj

                for srv_method, _ in inspect.getmembers(srv_class, predicate=inspect.isfunction):
                    if srv_method in method_services:
                        raise DuplicateMethodError("Multiple services implementing %s detected: %s and %s" % (srv_method, method_services[srv_method], srv_class))

                    method_services[srv_method] = srv_class
                    self.__dict__[srv_method] = getattr(obj, srv_method)

    def _register_subscribers(self, objects):
        subscribers = defaultdict(list)

        for obj in objects:
            for subscriber_class in _get_classes(type(obj), tickers.Subscriber):
                subscribers[subscriber_class].append(obj)

        return subscribers

    def get_subscribers(self, subscriber_cls):
        return self.subscribers[subscriber_cls]


## ============================================================================
#                              Internal tickers


class _AnyProblem(Problem):
    def __init__(self, problem):
        self.problem = problem
        self.simple = isinstance(problem, SimpleProblem)

    def parse_batch(self, batch, is_train):
        return self.problem.parse_batch(batch, is_train)

    def batch_counters(self, parsed_batch, is_train, **kwargs):
        if self.simple:
            return self.problem.loss(parsed_batch, is_train, **kwargs)
        else:
            return self.problem.batch_counters(parsed_batch, is_train, **kwargs)

    def loss_multibatch(self, counters, is_train):
        if self.simple:
            return tf.reduce_mean(counters)
        else:
            return self.problem.loss_multibatch(counters, is_train)

    def summary_multibatch(self, counters, prefix, is_train):
        if self.simple:
            return [tf.summary.scalar('%s/loss' % prefix, tf.reduce_mean(counters))]
        else:
            op = self.problem.summary_multibatch(counters, prefix, is_train)
            if not isinstance(op, (list, tuple)):
                op = [op]
            return op

    def params_summary(self):
        if self.simple:
            return []
        else:
            op = self.problem.params_summary()
            if not isinstance(op, (list, tuple)):
                op = [op]
            return op

    def make_feed_dict(self, batch):
        if self.simple:
            return super(_AnyProblem, self).make_feed_dict(batch)
        else:
            return self.problem.make_feed_dict(batch)

    def get_batch_cost_fn(self):
        if self.simple:
            return super(_AnyProblem, self).get_batch_cost_fn()
        else:
            return self.problem.get_batch_cost_fn()


# - _GlobalStep - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


class _GlobalStepService(
        DistributedTicker,
        GlobalStepService
        ):

    def __init__(self, tick_every_steps):
        assert len(tf.get_collection(tf.GraphKeys.GLOBAL_STEP)) == 0, "Global step already registered!"

        self.global_step = 0
        self.global_step_var = tf.get_variable(
            'global_step', [], tf.int64,
            initializer=tf.constant_initializer(0), trainable=False
            )

        self.batch_no = 0
        self.batch_no_var = tf.get_variable(
            'batch_no', [], tf.int64,
            initializer=tf.constant_initializer(0), trainable=False
            )

        self.tick_every_steps = tick_every_steps

        if self.tick_every_steps > 0:
            self.tick_no = 0
            self.tick_no_var = tf.get_variable(
                'tick_no', [], tf.int64,
                initializer=tf.constant_initializer(0), trainable=False
                )

        with tf.name_scope("step"):
            # In on_started global_step_ingraph and batch_no_ingraph should return current value
            self.global_step_ingraph = tf.placeholder_with_default(self.global_step_var, shape=[])
            self.batch_no_ingraph = tf.placeholder_with_default(self.batch_no_var, shape=[])

            tf.add_to_collection(tf.GraphKeys.GLOBAL_STEP, self.global_step_ingraph)

            if self.tick_every_steps > 0:
                self.tick_no_ingraph = tf.placeholder_with_default(self.tick_no_var, shape=[])
                tf.add_to_collection("TICK_NO", self.tick_no_ingraph)

        self.tick_every_steps = tick_every_steps

    def on_finished(self):
        tf.get_collection_ref(tf.GraphKeys.GLOBAL_STEP).clear()

    def on_train_batch_ingraph(self):
        with tf.name_scope("step"):
            if self.tick_every_steps == 0:
                return [
                    tf.assign(self.global_step_var, self.global_step_var + 1),
                    tf.assign(self.batch_no_var, self.batch_no_var + 1)
                ]
            else:
                is_it_time_yet = tf.equal(tf.mod(self.tick_no_var, self.tick_every_steps), self.tick_every_steps - 1)

                incr_global_step = tf.cond(is_it_time_yet,
                    lambda: tf.assign(self.global_step_var, self.global_step_var + 1),
                    lambda: tf.identity(self.global_step_var))

                incr_batch_no = tf.cond(is_it_time_yet,
                    lambda: tf.assign(self.batch_no_var, self.batch_no_var + 1),
                    lambda: tf.identity(self.batch_no_var))

                with tf.control_dependencies([incr_global_step, incr_batch_no]):
                    incr_tick_no = tf.assign(self.tick_no_var, self.tick_no_var + 1)

                return [
                    incr_global_step,
                    incr_batch_no,
                    incr_tick_no
                ]

    def after_train_batch(self, ingraph_result):
        if self.tick_every_steps == 0:
            self.global_step, self.batch_no = ingraph_result
        else:
            self.global_step, self.batch_no, self.tick_no = ingraph_result

    def get_batch_no(self):
        return self.batch_no

    def get_batch_no_ingraph(self):
        return self.batch_no_ingraph

    def get_global_step(self):
        return self.global_step

    def get_global_step_ingraph(self):
        return self.global_step_ingraph

    def set_global_step(self, global_step):
        with tf.name_scope("step"):
            self.global_step = tf.get_default_session().run(
                tf.assign(self.global_step_var, global_step))

# - _TrainTicker - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


class _TrainTicker(DistributedTicker, ModelService, TrainService, LearningRateService):

    def __init__(self, problem, algorithm, uploader):
        # Bind to learning rate multiplier here
        LearningRateService.__init__(self, algorithm.learning_rate)

        self.problem = _AnyProblem(problem)
        self.algorithm = algorithm

        with tf.name_scope("counters"):
            self.local_counters = nested_map(lambda t: tf.expand_dims(t, 0), self.problem.batch_counters(uploader.get_next(), True))

        with tf.name_scope("loss"):
            self.local_loss = self.problem.loss_multibatch(self.local_counters, True)

        with tf.name_scope("aggregate"):
            self.counters = nested_map(lambda t: lib.ops.mpi.allgather(t), self.local_counters)
            self.loss = lib.ops.mpi.allreduce(self.local_loss, name='TrainLoss')

        with tf.name_scope("update"):
            self.update_op = self.algorithm.create_update_ops(self.local_loss, self.loss)

    def on_train_batch_ingraph(self):
        return [self.local_loss, self.update_op, self.get_learning_rate_ingraph()]

    def after_train_batch(self, ingraph_result):
        # Print dot.
        lr = ingraph_result[-1]
        self.set_learning_rate(lr)
        print('.', end='', file=sys.stderr, flush=True)

    def get_problem(self):
        return self.problem

    def get_model(self, name):
        return self.problem.problem.models[name]

    def get_train_counters_ingraph(self):
        return self.counters

    def get_train_loss_ingraph(self):
        return self.loss
