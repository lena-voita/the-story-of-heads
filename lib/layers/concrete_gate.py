import tensorflow as tf
from warnings import warn
import lib


class ConcreteGate:
    """
    A gate made of stretched concrete distribution (using experimental Stretchable Concrete™)
    Can be applied to sparsify neural network activations or weights.
    Example usage: https://gist.github.com/justheuristic/1118a14a798b2b6d47789f7e6f511abd
    :param shape: shape of gate variable. can be broadcasted.
        e.g. if you want to apply gate to tensor [batch, length, units] over units axis,
        your shape should be [1, 1, units]
    :param temperature: concrete sigmoid temperature, should be in (0, 1] range
        lower values yield better approximation to actual discrete gate but train longer
    :param stretch_limits: min and max value of gate before it is clipped to [0, 1]
        min value should be negative in order to compute l0 penalty as in https://arxiv.org/pdf/1712.01312.pdf
        however, you can also use tf.nn.sigmoid(log_a) as regularizer if min, max = 0, 1
    :param l0_penalty: coefficient on the regularizer that minimizes l0 norm of gated value
    :param l2_penalty: coefficient on the regularizer that minimizes l2 norm of gated value
    :param eps: a small additive value used to avoid NaNs
    :param hard: if True, gates are binarized to {0, 1} but backprop is still performed as if they were concrete
    :param local_rep: if True, samples a different gumbel noise tensor for each sample in batch,
        by default, noise is sampled using shape param as size.

    """

    def __init__(self, name, shape, temperature=0.33, stretch_limits=(-0.1, 1.1),
                 l0_penalty=0.0, l2_penalty=0.0, eps=1e-6, hard=False, local_rep=False):
        self.name = name
        self.temperature, self.stretch_limits, self.eps = temperature, stretch_limits, eps
        self.l0_penalty, self.l2_penalty = l0_penalty, l2_penalty
        self.hard, self.local_rep = hard, local_rep
        with tf.variable_scope(name):
            self.log_a = lib.ops.get_model_variable("log_a", shape=shape)

    def __call__(self, values, is_train=None, axis=None, reg_collection=tf.GraphKeys.REGULARIZATION_LOSSES):
        """ applies gate to values, if is_train, adds regularizer to reg_collection """
        is_train = lib.layers.basic.is_dropout_enabled() if is_train is None else is_train
        gates = self.get_gates(is_train, shape=tf.shape(values) if self.local_rep else None)

        if self.l0_penalty != 0 or self.l2_penalty != 0:
            reg = self.get_penalty(values=values, axis=axis)
            tf.add_to_collection(reg_collection, tf.identity(reg, name='concrete_gate_reg'))
        return values * gates

    def get_gates(self, is_train, shape=None):
        """ samples gate activations in [0, 1] interval """
        low, high = self.stretch_limits
        with tf.name_scope(self.name):
            if is_train:
                shape = tf.shape(self.log_a) if shape is None else shape
                noise = tf.random_uniform(shape, self.eps, 1.0 - self.eps)
                concrete = tf.nn.sigmoid((tf.log(noise) - tf.log(1 - noise) + self.log_a) / self.temperature)
            else:
                concrete = tf.nn.sigmoid(self.log_a)

            stretched_concrete = concrete * (high - low) + low
            clipped_concrete = tf.clip_by_value(stretched_concrete, 0, 1)
            if self.hard:
                hard_concrete = tf.to_float(tf.greater(clipped_concrete, 0.5))
                clipped_concrete = clipped_concrete + tf.stop_gradient(hard_concrete - clipped_concrete)
        return clipped_concrete

    def get_penalty(self, values=None, axis=None):
        """
        Computes l0 and l2 penalties. For l2 penalty one must also provide the sparsified values
        (usually activations or weights) before they are multiplied by the gate
        Returns the regularizer value that should to be MINIMIZED (negative logprior)
        """
        if self.l0_penalty == self.l2_penalty == 0:
            warn("get_penalty() is called with both penalties set to 0")
        low, high = self.stretch_limits
        assert low < 0.0, "p_gate_closed can be computed only if lower stretch limit is negative"
        with tf.name_scope(self.name):
            # compute p(gate_is_closed) = cdf(stretched_sigmoid < 0)
            p_open = tf.nn.sigmoid(self.log_a - self.temperature * tf.log(-low / high))
            p_open = tf.clip_by_value(p_open, self.eps, 1.0 - self.eps)

            total_reg = 0.0
            if self.l0_penalty != 0:
                if values != None and self.local_rep:
                    p_open += tf.zeros_like(values)  # broadcast shape to account for values
                l0_reg = self.l0_penalty * tf.reduce_sum(p_open, axis=axis)
                total_reg += tf.reduce_mean(l0_reg)

            if self.l2_penalty != 0:
                assert values is not None
                l2_reg = 0.5 * self.l2_penalty * p_open * tf.reduce_sum(values ** 2, axis=axis)
                total_reg += tf.reduce_mean(l2_reg)

            return total_reg

    def get_sparsity_rate(self, is_train=False):
        """ Computes the fraction of gates which are now active (non-zero) """
        is_nonzero = tf.not_equal(self.get_gates(is_train), 0.0)
        return tf.reduce_mean(tf.to_float(is_nonzero))