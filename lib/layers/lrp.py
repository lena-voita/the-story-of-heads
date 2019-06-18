import tensorflow as tf
from ..ops import record_activations as rec


class LRP:
    """ Helper class for layerwise relevance propagation """
    alpha = 1.0
    beta = 0.0
    eps = 1e-7
    crop_function = abs

    @classmethod
    def relprop(cls, f_positive, f_negative, output_relevance, *inps):
        """
        computes input relevance given output_relevance using z+ rule
        works for linear layers, convolutions, poolings, etc.
        notation from DOI:10.1371/journal.pone.0130140, Eq 60
        :param f_positive: forward function with positive weights (if any) and no nonlinearities
        :param f_negative: forward function with negative weights and no nonlinearities
            if there's no weights, set f_negative to None. Only used for alpha-beta LRP
        :param output_relevance: relevance w.r.t. layer output
        :param inps: a list of layer inputs
        """
        assert len(inps) > 0, "please provide at least one input"
        with rec.do_not_record():
            alpha, beta, eps = cls.alpha, cls.beta, cls.eps
            inps = [inp + eps for inp in inps]

            # ouput relevance: [*dims, out_size]
            z_positive = f_positive(*inps)
            s_positive = cls.alpha * output_relevance / z_positive  # [*dims, out_size]
            positive_relevances = tf.gradients(z_positive, inps, grad_ys=s_positive)
            # ^-- list of [*dims, inp_size]

            if cls.beta != 0 and f_negative is not None:
                z_negative = f_negative(*inps)
                s_negative = -cls.beta * output_relevance / z_negative  # [*dims, out_size]
                negative_relevances = tf.gradients(z_negative, inps, grad_ys=s_negative)
                # ^-- list of [*dims, inp_size]
            else:
                negative_relevances = [0.0] * len(inps)

            inp_relevances = [
                inp * (rel_pos + rel_neg)
                for inp, rel_pos, rel_neg in zip(inps, positive_relevances, negative_relevances)
            ]

            return cls.rescale(output_relevance, *inp_relevances)


    @classmethod
    def rescale(cls, reference, *inputs,  axis=None):
        inputs = [cls.crop_function(inp) for inp in inputs]
        ref_scale = tf.reduce_sum(reference, axis=axis, keep_dims=axis is not None)
        inp_scales = [tf.reduce_sum(inp, axis=axis, keep_dims=axis is not None) for inp in inputs]
        total_inp_scale = sum(inp_scales) + cls.eps
        inputs = [inp * (ref_scale / total_inp_scale) for inp in inputs]
        return inputs[0] if len(inputs) == 1 else inputs
