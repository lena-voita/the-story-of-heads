import tensorflow as tf

import re
import sys
from fnmatch import fnmatch
from warnings import warn

from lib.train.tickers import LearningRateFn
from lib.ops import mpi
from . import optimizers


class GradientAccumulator:
    def __init__(self, var):
        self.target_var = var
        self.device = var.device

        with tf.device(self.device):
            self.accumulator_var = tf.get_variable(
                var.name[:-2] + '_grad',
                var.shape,
                var.dtype,
                initializer=tf.constant_initializer(0),
                trainable=False)

    def get_value(self, grad, n_steps, average):
        with tf.device(self.device):
            if average:
                return tf.div(self.accumulator_var + grad, n_steps)
            else:
                return self.accumulator_var + grad

    def update(self, grad):
        with tf.device(self.device):
            return tf.assign_add(self.accumulator_var, grad)

    def reset(self):
        with tf.device(self.device):
            return tf.assign(self.accumulator_var, tf.zeros(tf.shape(self.accumulator_var)))


class Algorithm(object):
    def __init__(self, learning_rate):
        if callable(learning_rate):
            self.learning_rate = learning_rate
        else:
            self.learning_rate = LearningRateFn('constant', learning_rate)
        self.learning_rate_ingraph = None

    @property
    def name(self):
        return "adam"

    def create_update_ops(self, local_loss, loss):
        raise NotImplementedError()

    def _make_learning_rate_ingraph(self):
        self.learning_rate_ingraph = self.learning_rate()


class GenericSgdAlgorithm(Algorithm):
    def __init__(self, learning_rate, dump_dir=None, dump_first_n=None, sync_every_steps=0,
                 variables=None, force_check_grad=True, average_grads=False, clip_norm=0, **kwargs):
        Algorithm.__init__(self, learning_rate)
        self.dump_dir = dump_dir
        self.dump_first_n = dump_first_n
        self.sync_every_steps = sync_every_steps
        self.average_grads = average_grads
        self.force_check_grad = force_check_grad
        self.clip_norm = clip_norm

        self.var_list = (
            tf.trainable_variables() +
            tf.get_collection(tf.GraphKeys.TRAINABLE_RESOURCE_VARIABLES))

        if isinstance(variables, str):
            variables = [variables]
            print("WARNING: Variables should be list, not a string!", file=sys.stderr, flush=True)

        if variables:

            for m in variables:
                assert any(fnmatch(v.name.split(':')[0], m) for v in self.var_list), \
                    "Pattern '{}' does not match any variables!".format(m)

            self.var_list = [v for v in self.var_list if any(fnmatch(v.name.split(':')[0], m) for m in variables)]

        print("Variables to optimize: %s" % ', '.join(v.name for v in self.var_list), file=sys.stderr, flush=True)

    def create_update_ops(self, local_loss, loss):
        self._make_learning_rate_ingraph()
        with tf.variable_scope(self.name):
            optimizer = self._create_optimizer()

            grads = tf.gradients(
                local_loss, self.var_list,
                colocate_gradients_with_ops=True,
                gate_gradients=1)

            # handle None grads
            for grad, var in zip(grads, self.var_list):
                if grad is None:
                    if self.force_check_grad:
                        raise ValueError("Gradient for %s is None. Make sure loss is differentiable "
                                         "w.r.t. %s or set force_check_grad=False in optimizer_opts" % (var.name, var.name))
                    else:
                        warn("Gradient for %s is None. It will be ignored!" % var.name)

            grads, self.var_list = zip(*[(grad, var) for grad, var in zip(grads, self.var_list)
                                            if grad is not None])

            if self.sync_every_steps > 0:
                with tf.variable_scope(self.name):
                    self.grads_acc = [GradientAccumulator(v) for v in self.var_list]

            if self.sync_every_steps == 0:
                return self._compute_update_op(optimizer, grads, loss)
            else:
                # sync every N steps
                global_step = tf.get_collection("TICK_NO")[0]
                is_it_time_yet = tf.equal(tf.mod(global_step, self.sync_every_steps), self.sync_every_steps - 1)

                # compute update op or fake update op
                update_op = tf.cond(is_it_time_yet,
                    lambda: self._compute_update_op(
                        optimizer,
                        [a.get_value(g, self.sync_every_steps, self.average_grads) for a, g in zip(self.grads_acc, grads)],
                        loss),
                    lambda: tf.no_op()
                )

                with tf.control_dependencies([update_op]), tf.name_scope('accumulate'):
                    assign_op = tf.cond(is_it_time_yet,
                                        lambda: [tf.group(a.reset()) for a in self.grads_acc],
                                        lambda: [tf.group(a.update(g)) for a, g in zip(self.grads_acc, grads)])

                return update_op, assign_op

    def _compute_update_op(self, optimizer, grads, loss):
        with tf.name_scope("aggregate"):
            # Gradient clipping.
            if self.clip_norm > 0:
                with tf.name_scope("clip"):
                    grads = [tf.clip_by_norm(grad, self.clip_norm) for grad in grads]

            # Reduce gradients
            grads = [mpi.allreduce(grad, name=re.sub('\\W', '_', var.name) + '_allreduce')
                     for grad, var in zip(grads, self.var_list)]

            with tf.name_scope("apply"):
                update_op, var_update_ops = optimizer.apply_gradients(zip(grads, self.var_list), loss)

        return update_op

    def _create_optimizer(self):
        raise NotImplementedError()


class Sgd(GenericSgdAlgorithm):

    @property
    def name(self):
        return "sgd"

    def _create_optimizer(self):
        return optimizers.GradientDescentOptimizer(self.learning_rate_ingraph)


class RMSProp(GenericSgdAlgorithm):

    def __init__(self, learning_rate, clip_norm=0, variables=None,
                 sync_every_steps=0, average_grads=False, force_check_grad=True, **kwargs):
        GenericSgdAlgorithm.__init__(self, learning_rate=learning_rate, variables=variables, clip_norm=clip_norm,
                                     sync_every_steps=sync_every_steps, average_grads=average_grads,
                                     force_check_grad=force_check_grad)
        self.kwargs = kwargs

    @property
    def name(self):
        return "rms_prop"

    def _create_optimizer(self):
        return optimizers.RMSPropOptimizer(self.learning_rate_ingraph, **self.kwargs)


class Adam(GenericSgdAlgorithm):

    def __init__(self, learning_rate, clip_norm=0, variables=None, dump_dir=None, dump_first_n=None,
                 sync_every_steps=0, average_grads=False, force_check_grad=True, *args, **kwargs):
        GenericSgdAlgorithm.__init__(self, learning_rate=learning_rate, dump_dir=dump_dir, dump_first_n=dump_first_n,
                                     clip_norm=clip_norm, sync_every_steps=sync_every_steps, variables=variables,
                                     average_grads=average_grads, force_check_grad=force_check_grad)
        self.args = args
        self.kwargs = kwargs

    @property
    def name(self):
        return "adam"

    def _create_optimizer(self):
        return optimizers.AdamOptimizer(self.learning_rate_ingraph, *self.args, **self.kwargs)


class Eve(GenericSgdAlgorithm):

    def __init__(self, learning_rate, clip_norm=0, variables=None, sync_every_steps=0, average_grads=False,
                 force_check_grad=True, *args, **kwargs):
        GenericSgdAlgorithm.__init__(self, learning_rate=learning_rate, clip_norm=clip_norm, variables=variables,
                                     sync_every_steps=sync_every_steps, average_grads=average_grads,
                                     force_check_grad=force_check_grad)
        self.args = args
        self.kwargs = kwargs

    @property
    def name(self):
        return "eve"

    def _create_optimizer(self):
        return optimizers.EveOptimizer(self.learning_rate_ingraph, *self.args, **self.kwargs)


class LazyAdam(GenericSgdAlgorithm):
    def __init__(self, learning_rate, clip_norm=0, variables=None, dump_dir=None, dump_first_n=None,
                 sync_every_steps=0, average_grads=False, force_check_grad=True, *args, **kwargs):
        GenericSgdAlgorithm.__init__(self, learning_rate=learning_rate, clip_norm=clip_norm, variables=variables,
                                     sync_every_steps=sync_every_steps, average_grads=average_grads,
                                     force_check_grad=force_check_grad, dump_dir=dump_dir, dump_first_n=dump_first_n)
        self.args = args
        self.kwargs = kwargs

    @property
    def name(self):
        return "lazy_adam"

    def _create_optimizer(self):
        return optimizers.LazyAdamOptimizer(self.learning_rate_ingraph, *self.args, **self.kwargs)
