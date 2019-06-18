# Gradient optimizers


import tensorflow as tf
from .. import ops

## ----------------------------------------------------------------------------
#                                  Base


class OptimizerBase:
    def apply_gradients(self, grads_and_vars, loss):
        """
        Create internal optimizer state and return an operation that performs
        one optimization step.

        Ret: complete_update_op, per_var_update_ops.

        'complete_update_op': single operation that performs one optimization
        step and updates everything -- model parameters and internal optimizer
        state.

        'per_var_update_ops': list of operations, one per model variable; each
        operation assigns its respective variable the new value. Order of
        per_var_update_ops matches the order of grads_and_vars.
        """
        raise NotImplementedError()


## ----------------------------------------------------------------------------
#                                  SGD


class GradientDescentOptimizer(OptimizerBase):
    def __init__(self, learning_rate):
        self._learning_rate = learning_rate

    def apply_gradients(self, grads_and_vars, loss):
        spy = _GradientDescentOptimizerSpy(self._learning_rate)
        return spy.get_update_ops(grads_and_vars)


class _GradientDescentOptimizerSpy(tf.train.GradientDescentOptimizer):
    """
    Like tf.train.GradientDescentOptimizer, but is able to return per-variable
    update operations along with operation that does the complete update
    """

    def get_update_ops(self, grads_and_vars):
        _complete_update_op = super().apply_gradients(grads_and_vars)
        return _complete_update_op, self._per_var_update_ops

    def _finish(self, update_ops, name_scope):
        self._per_var_update_ops = update_ops
        return super()._finish(update_ops, name_scope)


## ----------------------------------------------------------------------------
#                                  RMSProp


class RMSPropOptimizer(OptimizerBase):
    def __init__(self, learning_rate, **kwargs):
        self._learning_rate = learning_rate
        self._kwargs = kwargs

    def apply_gradients(self, grads_and_vars, loss):
        spy = _RMSPropOptimizerSpy(self._learning_rate, **self._kwargs)
        return spy.get_update_ops(grads_and_vars)


class _RMSPropOptimizerSpy(tf.train.RMSPropOptimizer):
    """
    Like tf.train.RMSPropOptimizerOptimizer, but is able to return per-variable
    update operations along with operation that does the complete update
    """

    def get_update_ops(self, grads_and_vars):
        _complete_update_op = super().apply_gradients(grads_and_vars)
        return _complete_update_op, self._per_var_update_ops

    def _finish(self, update_ops, name_scope):
        self._per_var_update_ops = update_ops
        return super()._finish(update_ops, name_scope)


## ----------------------------------------------------------------------------
#                                   Adam


class AdamOptimizer(OptimizerBase):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self._learning_rate = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon

    def apply_gradients(self, grads_and_vars, loss):
        spy = _AdamOptimizerSpy(
            self._learning_rate,
            self._beta1,
            self._beta2,
            self._epsilon
            )
        return spy.get_update_ops(grads_and_vars)


class _AdamOptimizerSpy(tf.train.AdamOptimizer):
    """
    Like tf.train.AdamOptimizer, but is able to return per-variable update
    operations along with operation that does the complete update
    """

    def get_update_ops(self, grads_and_vars):
        _complete_update_op = super().apply_gradients(grads_and_vars)
        return _complete_update_op, self._per_var_update_ops

    def _finish(self, update_ops, name_scope):
        self._per_var_update_ops = update_ops
        return super()._finish(update_ops, name_scope)


## ----------------------------------------------------------------------------
#                                   LazyAdam


class LazyAdamOptimizer(OptimizerBase):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.98, epsilon=1e-9):
        self._learning_rate = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon

    def apply_gradients(self, grads_and_vars, loss):
        spy = _LazyAdamOptimizerSpy(
            self._learning_rate,
            self._beta1,
            self._beta2,
            self._epsilon
            )
        return spy.get_update_ops(grads_and_vars)


class _LazyAdamOptimizerSpy(tf.contrib.opt.LazyAdamOptimizer):
    """
    Like tf.contrib.opt.LazyAdamOptimizer, but is able to return per-variable
    update operations along with operation that does the complete update
    """

    def get_update_ops(self, grads_and_vars):
        _complete_update_op = super().apply_gradients(grads_and_vars)
        return _complete_update_op, self._per_var_update_ops

    def _finish(self, update_ops, name_scope):
        self._per_var_update_ops = update_ops
        return super()._finish(update_ops, name_scope)


## ----------------------------------------------------------------------------
#                                   Eve


class EveOptimizer(OptimizerBase):
    def __init__(
            self, learning_rate=0.001, threshold_lower=0.1, threshold_upper=10,
            beta1=0.9, beta2=0.999, beta3=0.999
            ):

        self._lr = learning_rate
        self._threshold_lower = threshold_lower
        self._threshold_upper = threshold_upper
        self._beta1 = beta1
        self._beta2 = beta2
        self._beta3 = beta3

    def apply_gradients(self, grads_and_vars, loss):
        grads_and_vars = list(grads_and_vars)
        with tf.variable_scope("eve"):
            return self._apply_gradients(grads_and_vars, loss)

    def _apply_gradients(self, grads_and_vars, loss):
        # 1. Create scalar variables
        d = tf.get_variable(
            "d",
            shape=[1],
            initializer=tf.constant_initializer(1.),
            trainable=False
            )
        f_hat = tf.get_variable(
            "f_hat",
            shape=[],
            dtype = tf.float32,
            initializer=tf.constant_initializer(0.),
            trainable=False
            )
        t = tf.get_variable(
            "t",
            shape=[],
            initializer=tf.constant_initializer(1.),
            trainable=False
            )

    # 2. Create tensor variables.
        v = []
        m = []
        for _, var in grads_and_vars:
            with tf.variable_scope("v"):
                v.append(tf.get_variable(
                    var.name.replace(':', ''),
                    initializer=tf.zeros(var.get_shape()),
                    trainable=False
                    ))
            with tf.variable_scope("m"):
                m.append(tf.get_variable(
                    var.name.replace(':', ''),
                    initializer=tf.zeros(var.get_shape()),
                    trainable=False
                    ))

        # 3. Create conditional updates.
        def update_not_at_start():
            loss_increased = tf.greater_equal(loss, f_hat)
            delta1_t = tf.cond(
                loss_increased,
                lambda: tf.constant(self._threshold_lower + 1.),
                lambda: tf.constant(1. / (self._threshold_upper+1.))
                )
            delta2_t = tf.cond(
                loss_increased,
                lambda: tf.constant(self._threshold_upper + 1.),
                lambda: tf.constant(1. / (self._threshold_lower+1.))
                )
            c_t = tf.minimum(tf.maximum(delta1_t, loss/f_hat), delta2_t)
            r_t = tf.abs((c_t-1.)*f_hat) / tf.minimum(c_t*f_hat, f_hat)
            d_t = self._beta3*d + (1.-self._beta3)*r_t
            return ops.group(
                d.assign(d_t).op,
                f_hat.assign(c_t * f_hat).op
                )

        def update_at_start():
            return ops.group(f_hat.assign(loss).op)

        at_start = tf.greater_equal(t, 2.)
        updates = [tf.cond(at_start, update_not_at_start, update_at_start).op]

        # 4. Create unconditional updates.

        # Updates for trainable variables.
        per_var_update_ops = []
        for i in range(len(grads_and_vars)):
            g = grads_and_vars[i][0]
            p = grads_and_vars[i][1]

            if self._beta1 > 0:
                m_t = self._beta1*m[i] + tf.scalar_mul(1.-self._beta1, g)
                m_hat = m_t / (1. - tf.pow(self._beta1, t))
                updates.append(m[i].assign(m_t).op)
            else:
                m_hat = g

            v_t = self._beta2*v[i] + (1.-self._beta2)*tf.square(g)
            v_hat = v_t / (1. - tf.pow(self._beta2, t))
            p_t = p - self._lr * (m_hat / (d * tf.sqrt(v_hat) + 1e-8))

            updates.append(v[i].assign(v_t).op)
            per_var_update_ops.append(p.assign(p_t).op)

        updates.append(t.assign(t + 1.).op)

        return tf.group(*(updates + per_var_update_ops)), per_var_update_ops
