# Model definition


## ============================================================================
#                               SimpleProblem


class SimpleProblem:
    """
    SimpleProblem is an interace which describes gradient minimization problem.

    You might wanna take a look at Model interface if you're doing more than a
    simple demonstration. Some use cases for Model are:

        - Account for number of items in batch (to implement proper averaging)
        - Print extra summary in tensorboard
    """

    def loss(self, batch, is_train):
        """
        Return batch-average loss, a scalar Tensor.

        'batch' is a nested structure that has the same schema as input
        data, except values are now Tensors. For example, if training data is

            { 'inp': np.array([1, 2, 3]),
              'out': np.array([3, 2, 1]) }

        then 'batch' will be

            { 'inp': tf.Tensor(...),
              'out': tf.Tensor(...) }

        trainer performs asynchronous data upload to tensorflow, so the tensors point to internal tensorflow buffers.

        loss() on devset is averaged among devset batches. If you want
        averaging with proper account of number of items in every batch, take
        a look at Model class.

        'is_train' specifies whether the the function should compute trainset
        or devset loss.
        """
        raise NotImplementedError()


## ============================================================================
#                                    Problem


class Problem:
    """
    Like SimpleProblem, but with a more sophisticated interface: instead of loss(), you
    must override batch_counters() and loss_multibatch().

    Values of batch_counters() are accumulated to a Tensor among a new 0'th
    axis, so that loss_multibatch() can do custom aggregation.
    """
    def batch_counters(self, batch, is_train):
        """
        'batch': same as in SimpleModel.loss()

        Return nested structure of Tensors for use in loss_multibatch() and
        summary_multibatch().
        """
        raise NotImplementedError()

    def loss_multibatch(self, counters, is_train):
        """
        Aggregate 'counters' into single scalar loss Tensor.

        'counters': nested structure of Tensors returned by batch_counters().
        Each Tensor in the nested structure has an additional 0'th axis, each
        element holding a result of batch_counters() invocation.

        'is_train' specifies whether the the function should compute trainset
        or devset loss.
        """
        raise NotImplementedError()

    def summary_multibatch(self, counters, prefix, is_train):
        """
        Return operation or a list of operations that compute aggregated
        summary for multiple batches.

        Every returned summary operation should have given 'prefix'.

        'is_train' specifies whether the the function should compute trainset
        or devset summary.
        """
        return []

    def params_summary(self):
        """
        Return an operation or a list of operations that compute custom
        summaries of model parameters.

        Note that training automatically computes parameter histogram summaries,
        so you don't need to do that yourself.
        """
        return []
