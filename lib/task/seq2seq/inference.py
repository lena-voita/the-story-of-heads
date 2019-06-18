import sys
from collections import namedtuple
from warnings import warn

import tensorflow as tf

import lib.util
from lib.ops import infer_length, infer_mask
from lib.ops.sliced_argmax import sliced_argmax
from lib.util import nested_map, is_scalar
import numpy as np


def translate_lines(lines, translator, model, out_voc, replace_unk=False, unbpe=False, dumper=None):
    """
    tokenize, translate and detokenize strings using specified model and translator
    :param lines: an iterable of strings
    :type translator: something that can .translate_batch(batch_dict) -> out, attnP, ...
    :param model: a model from lib.task.seq2seq.models.ModelBase
    :param out_voc: destination language dictionary
    :param replace_unk: if True, forbids sampling UNK from the model
    :param unbpe: if True, concatenates bpe subword units together
    :return: yields one translation line at a time
    """
    batch = [(l, "") for l in lines]
    batch_data = model.make_feed_dict(batch, add_inp_words=True)
    kwargs = {}

    if dumper is not None:
        kwargs['batch_dumpers'] = dumper.create_batch_dumpers(batch)

    out_ids, attnP = translator.translate_batch(batch_data, **kwargs)[:2]

    for i in range(len(out_ids)):
        ids = list(out_ids[i])
        words = out_voc.words(ids)
        words = [w for w, out_id in zip(words, ids) if out_id not in [out_voc.bos, out_voc.eos]]

        if replace_unk:
            where = [(w and '_UNK_' in w) for w in words]
            if any(w for w in where):
                inp_words = batch_data['inp_words'][i][:batch_data['inp_len'][i]]

                # select attention weights for non-BOS/EOS tokens, shape=[num_outputs, num_inputs]
                attns = np.array([a for a, out_id in zip(attnP[i], ids)
                                  if out_id not in [out_voc.bos, out_voc.eos]])[:len(words), :len(inp_words)]

                # forbid attns to special tokens if there are normal tokens in inp
                inp_mask = np.array([w not in ['_BOS_', '_EOS_'] for w in inp_words])
                attns = np.where(inp_mask[None, :], attns, -np.inf)

                words = copy_argmax(inp_words, words, attns, where)

        out_line = " ".join(words)
        if unbpe:
            out_line = out_line.replace('` ', '')
        yield out_line


def copy_argmax(inp, out, attnP, where):
    """
    inp: [ninp]
    out: [nout]
    attnP: [nout, ninp]
    where: [nout]
    """
    # Check shapes.
    if len(inp) != attnP.shape[1]:
        msg = 'len(inp) is %i, but attnP.shape[1] is %i'
        raise ValueError(msg % (len(inp), attnP.shape[1]))
    if len(out) != attnP.shape[0]:
        msg = 'len(out) is %i, but attnP.shape[0] is %i'
        raise ValueError(msg % (len(out), attnP.shape[0]))

    # Copy in every requested position.
    new_out = []
    for o in range(len(out)):
        # Output as-is.
        if not where[o]:
            new_out.append(out[o])
            continue

        # Copy from input.
        i = np.argmax(attnP[o])
        new_out.append(inp[i])

    return new_out


class TranslateModel:

    def __init__(self, name, inp_voc, out_voc, loss, **hp):
        """ Each model must have name, vocabularies and a hyperparameter dict """
        self.name = name
        self.inp_voc = inp_voc
        self.out_voc = out_voc
        self.loss = loss
        self.hp = hp

    def encode(self, batch, **flags):
        """
        Encodes symbolic input and returns initial state of decode
        :param batch: {
            inp: int32 matrix [batch,time] or whatever your model can encode
            inp_len: int vector [batch_size]
        }
        --------------------------------------------------
        :returns: dec_state, nested structure of tensors, batch-major
        """
        raise NotImplementedError()

    def decode(self, dec_state, words, **flags):
        """
        Performs decode step on given words.

        dec_state: nested structure of tensors, batch-major
        words: int vector [batch_size]
        ------------------------------------------------------
        :returns: new_dec_state, nested structure of tensors, batch-major
        """
        raise NotImplementedError()

    def shuffle(self, dec_state, hypo_indices):
        """
        Selects hypotheses from model decoder state by given indices.
        :param dec_state: a nested structure of tensors representing model state
        :param hypo_indices: int32 vector of indices to select
        :returns: dec state elements for given flat_indices only
        """
        return nested_map(lambda x: tf.gather(x, hypo_indices), dec_state)

    def switch(self, condition, state_on_true, state_on_false):
        """
        Composes a new stack.best_dec_state out of new dec state when new_is_better and old dec state otherwise
        :param condition: a boolean condition vector of shape [batch_size]
        """
        return nested_map(lambda x, y: tf.where(condition, x, y), state_on_true, state_on_false)

    def sample(self, dec_state, base_scores, slices, k, sampling_strategy='greedy', sampling_temperature=None, **flags):
        """
        Samples top-K new words for each hypothesis from a beam.
        Decoder states and base_scores of hypotheses for different inputs are concatenated like this:
            [x0_hypo0, x0_hypo1, ..., x0_hypoN, x1_hypo0, ..., x1_hypoN, ..., xM_hypoN

        :param dec_state: nested structure of tensors, batch-major
        :param base_scores: [batch_size], log-probabilities of hypotheses in dec_state with additive penalties applied
        :param slices: start indices of each input
        :param k: [], int, how many hypotheses to sample per input
        :returns: best_hypos, words, scores,
            best_hypos: in-beam hypothesis index for each sampled token, [batch_size / slice_size, k], int
            words: new tokens for each hypo, [batch_size / slice_size, k], int
            scores: log P(words | best_hypos), [batch_size / slice_size, k], float32
        """
        rdo = self.get_rdo(dec_state)
        if isinstance(rdo, (tuple, list)) or lib.util.is_namedtuple(rdo):
            logits = self.loss.rdo_to_logits__predict(*rdo)
        else:
            logits = self.loss.rdo_to_logits__predict(rdo)

        n_hypos, voc_size = tf.shape(logits)[0], tf.shape(logits)[1]
        batch_size = tf.shape(slices)[0]

        if sampling_strategy == 'random':
            if sampling_temperature is not None:
                logits /= sampling_temperature

            logp = tf.nn.log_softmax(logits, 1)

            best_hypos = tf.range(0, n_hypos)[:, None]

            best_words = tf.cast(tf.multinomial(logp, k), tf.int32)
            best_words_flat = (tf.range(0, batch_size) * voc_size)[:, None] + best_words

            best_delta_scores = tf.gather(tf.reshape(logp, [-1]), best_words_flat)

        elif sampling_strategy == 'greedy':
            logp = tf.nn.log_softmax(logits, 1) + base_scores[:, None]
            best_scores, best_indices = sliced_argmax(logp, slices, k)

            # If best_hypos == -1, best_scores == -inf, set best_hypos to 0 to avoid runtime IndexError
            best_hypos = tf.where(tf.not_equal(best_indices, -1),
                                  tf.floordiv(best_indices, voc_size) + slices[:, None],
                                  tf.fill(tf.shape(best_indices), -1))
            best_words = tf.where(tf.not_equal(best_indices, -1),
                                  tf.mod(best_indices, voc_size),
                                  tf.fill(tf.shape(best_indices), -1))

            best_delta_scores = best_scores - tf.gather(base_scores, tf.maximum(0, best_hypos))
        else:
            raise ValueError("sampling_strategy must be in ['random','greedy']")

        return (best_hypos, best_words, best_delta_scores)

    def get_rdo(self, dec_state):
        if hasattr(dec_state, 'rdo'):
            return dec_state.rdo
        raise NotImplementedError()

    def get_attnP(self, dec_state):
        """
        Returns attnP

        dec_state: [..., batch_size, ...]
        ---------------------------------
        Ret: attnP
            attnP: [batch_size, ninp]
        """
        if hasattr(dec_state, 'attnP'):
            return dec_state.attnP
        raise NotImplementedError()


class GreedyDecoder:
    """
    Inference that encodes input sequence, then iteratively samples and decodes output sequence.
    :type model: lib.task.seq2seq.inference.translate_model.TranslateModel
    :param batch: a dictionary that contains symbolic tensor {'inp': input token ids, shape [batch_size,time]}
    :param max_len: maximum length of output sequence, symbolic or numeric integer
        if scalar, sets global length; if vector[batch_size], sets length for each input;
        if None, defaults to 2*inp_len + 3
    :param force_bos: if True, forces zero-th output to be model.out_voc.bos. Otherwise lets model decide.
    :param force_eos: if True, any token past initial EOS is guaranteed to be EOS
    :param get_tracked_outputs: callback that returns whatever tensor(s) you want to track on each time-step
    :param crop_last_step: if True, does not perform  additional decode __after__ last eos
            ensures all tensors have equal time axis
    :param back_prop: see tf.while_loop back_prop param
    :param swap_memory: see tf.while_loop swap_memory param
    :param **flags: you can add any amount of tags that encode and decode understands.
        e.g. greedy=True or is_train=True

    """

    Stack = namedtuple('Stack',
                       ['out', 'out_len', 'scores', 'finished', 'dec_state', 'attnP', 'tracked'])

    def __init__(self, model, batch_placeholder, max_len=None, force_bos=False, force_eos=True,
                 get_tracked_outputs=lambda dec_state: [], crop_last_step=True,
                 back_prop=True, swap_memory=False, **flags):
        self.batch_placeholder = batch_placeholder
        self.get_tracked_outputs = get_tracked_outputs

        inp_len = batch_placeholder.get('inp_len', infer_length(batch_placeholder['inp'], model.out_voc.eos))
        max_len = max_len if max_len is not None else (2 * inp_len + 3)

        first_stack = self.create_initial_stack(model, batch_placeholder, force_bos=force_bos, **flags)
        shape_invariants = nested_map(lambda v: tf.TensorShape([None for _ in v.shape]), first_stack)

        # Actual decoding
        def should_continue_translating(*stack):
            stack = self.Stack(*stack)
            return tf.reduce_any(tf.less(stack.out_len, max_len)) & tf.reduce_any(~stack.finished)

        def inference_step(*stack):
            stack = self.Stack(*stack)
            return self.greedy_step(model, stack, **flags)

        final_stack = tf.while_loop(
            cond=should_continue_translating,
            body=inference_step,
            loop_vars=first_stack,
            shape_invariants=shape_invariants,
            swap_memory=swap_memory,
            back_prop=back_prop,
        )

        outputs, _, scores, _, dec_states, attnP, tracked_outputs = final_stack
        if crop_last_step:
            attnP = attnP[:, :-1]
            tracked_outputs = nested_map(lambda out: out[:, :-1], tracked_outputs)

        if force_eos:
            out_mask = infer_mask(outputs, model.out_voc.eos)
            outputs = tf.where(out_mask, outputs, tf.fill(tf.shape(outputs), model.out_voc.eos))

        self.best_out = outputs
        self.best_attnP = attnP
        self.best_scores = scores
        self.dec_states = dec_states
        self.tracked_outputs = tracked_outputs

    def translate_batch(self, batch_data, **optional_feed):
        """
        Translates NUMERIC batch of data
        :param batch_data: dict {'inp':np.array int32[batch,time]}
        :optional_feed: any additional values to be fed into graph. e.g. if you used placeholder for max_len at __init__
        :return: best hypotheses' outputs[batch, out_len] and attnP[batch, out_len, inp_len]
        """
        feed_dict = {placeholder: batch_data[k] for k, placeholder in self.batch_placeholder.items()}
        for k, v in optional_feed.items():
            feed_dict[k] = v

        out_ids, attnP = tf.get_default_session().run(
            [self.best_out, self.best_attnP],
            feed_dict=feed_dict)

        return out_ids, attnP

    def create_initial_stack(self, model, batch_placeholder, force_bos=False, **flags):
        inp = batch_placeholder['inp']
        batch_size = tf.shape(inp)[0]

        initial_state = model.encode(batch_placeholder, **flags)
        initial_attnP = model.get_attnP(initial_state)[:, None]
        initial_tracked = nested_map(lambda x: x[:, None], self.get_tracked_outputs(initial_state))

        if force_bos:
            initial_outputs = tf.cast(tf.fill((batch_size, 1), model.out_voc.bos), inp.dtype)
            initial_state = model.decode(initial_state, initial_outputs[:, 0], **flags)
            second_attnP = model.get_attnP(initial_state)[:, None]
            initial_attnP = tf.concat([initial_attnP, second_attnP], axis=1)
            initial_tracked = nested_map(lambda x, y: tf.concat([x, y[:, None]], axis=1),
                                         initial_tracked,
                                         self.get_tracked_outputs(initial_state),)
        else:
            initial_outputs = tf.zeros((batch_size, 0), dtype=inp.dtype)

        initial_scores = tf.zeros([batch_size], dtype='float32')
        initial_finished = tf.zeros_like([batch_size], dtype='bool')
        initial_len = tf.shape(initial_outputs)[1]

        return self.Stack(initial_outputs, initial_len, initial_scores, initial_finished,
                          initial_state, initial_attnP, initial_tracked)

    def greedy_step(self, model, stack, **flags):
        """
        :type model: lib.task.seq2seq.inference.translate_model.TranslateModel
        :param stack: beam search stack
        :return: new beam search stack
        """
        out_seq, out_len, scores, finished, dec_states, attnP, tracked = stack

        # 1. sample
        batch_size = tf.shape(out_seq)[0]
        phony_slices = tf.range(batch_size)
        _, new_outputs, logp_next = model.sample(dec_states, scores, phony_slices, k=1, **flags)

        out_seq = tf.concat([out_seq, new_outputs], axis=1)
        scores = scores + logp_next[:, 0] * tf.cast(~finished, 'float32')
        is_eos = tf.equal(new_outputs[:, 0], model.out_voc.eos)
        finished = tf.logical_or(finished, is_eos)

        # 2. decode
        new_states = model.decode(dec_states, new_outputs[:, 0], **flags)
        attnP = tf.concat([attnP, model.get_attnP(new_states)[:, None]], axis=1)
        tracked = nested_map(lambda seq, new: tf.concat([seq, new[:, None]], axis=1),
                             tracked, self.get_tracked_outputs(new_states)
                             )
        return self.Stack(out_seq, out_len + 1, scores, finished, new_states, attnP, tracked)


class BeamSearchDecoder:
    """
    Performs ingraph beam search for given input sequences (inp)
    Supports custom penalizing, pruning against best score and best score in beam (via beam_spread)
    :param model: something that implements TranslateModel
    :param batch_placeholder: whatever model can .encode,
        by default should be {'inp': int32 matrix [batch_size x time]}
    :param max_len: maximum hypothesis length to consider, symbolic or numeric integer
        if scalar, sets global length; if vector[batch_size], sets length for each input;
        if None, defaults to 2*inp_len + 3; float('inf') means unlimited
    :param min_len: minimum valid output length. None means min_len=inp_len // 4 - 1; Same type as min_len
    :param beam_size: maximum number of hypotheses that can pass from one beam search step to another.
        The rest is pruned.
    :param beam_spread: maximum difference in score between a hypothesis and current best hypothesis.
        Anything below that is pruned.
    :param force_bos: if True, forces zero-th output to be model.out_voc.bos. Otherwise lets model decide.
    :param if_no_eos: if 'last', will return unfinished hypos if there are no finished hypos by max_len
                      elif 'initial', returns empty hypothesis
    :param back_prop: see tf.while_loop back_prop param
    :param swap_memory: see tf.while_loop swap_memory param

    :param **flags: whatever else you want to feed into model. This will be passed to encode, decode, etc.
        is_train - if True (default), enables dropouts and similar training-only stuff
        sampling_strategy - if "random", samples hypotheses proportionally to softmax(logits)
                              otherwise(default) - takes top K hypotheses
        sampling_temperature -  if sampling_strategy == "random",
            performs sampling ~ softmax(logits/sampling_temperature)

    """
    Stack = namedtuple('Stack', [
        # per hypo values
        'out',  # [batch_size x beam_size, nout], int
        'scores',  # [batch_size x beam_size ]
        'raw_scores',  # [batch_size x beam_size ]
        'attnP',  # [batch_size x beam_size, nout+1, ninp]
        'dec_state', # TranslateModel DecState nested structure of [batch_size x beam_size, ...]

        # per beam values
        'slices',  # indices of first hypo for each sentence [batch_size ]
        'out_len',  # total (maximum) length of a stack [], int
        'best_out',  # [batch_size, nout], int, padded with EOS
        'best_scores',  # [batch_size]
        'best_raw_scores',  # [batch_size]
        'best_attnP',  # [batch_size, nout+1, ninp], padded with EOS
        'best_dec_state', # TranslateModel DecState; nested structure of [batch_size, ...]

        # Auxilary data for extension classes.
        'ext' # Dict[StackExtType, StackExtType()]
        ])

    def __init__(self, model, batch_placeholder, min_len=None, max_len=None,
                 beam_size=12, beam_spread=3, beam_spread_raw=None, force_bos=False,
                 if_no_eos='last', back_prop=True, swap_memory=False, **flags
                 ):
        assert if_no_eos in ['last', 'initial']
        assert np.isfinite(beam_spread) or max_len != float('inf'), "Must set maximum length if beam_spread is infinite"
        # initialize fields
        self.batch_placeholder = batch_placeholder
        inp_len = batch_placeholder.get('inp_len', infer_length(batch_placeholder['inp'], model.out_voc.eos))
        self.min_len = min_len if min_len is not None else inp_len // 4 - 1
        self.max_len = max_len if max_len is not None else 2 * inp_len + 3
        self.beam_size, self.beam_spread = beam_size, beam_spread
        if beam_spread_raw is None:
            self.beam_spread_raw = beam_spread
        else:
            self.beam_spread_raw = beam_spread_raw
        self.force_bos, self.if_no_eos = force_bos, if_no_eos

        # actual beam search
        first_stack = self.create_initial_stack(model, batch_placeholder, force_bos=force_bos, **flags)
        shape_invariants = nested_map(lambda v: tf.TensorShape([None for _ in v.shape]), first_stack)

        def should_continue_translating(*stack):
            stack = self.Stack(*stack)
            should_continue = self.should_continue_translating(model, stack)
            return tf.reduce_any(should_continue)

        def expand_hypos(*stack):
            return self.beam_search_step(model, self.Stack(*stack), **flags)

        last_stack = tf.while_loop(
            cond=should_continue_translating,
            body=expand_hypos,
            loop_vars=first_stack,
            shape_invariants=shape_invariants,
            back_prop=back_prop,
            swap_memory=swap_memory,
        )

        # crop unnecessary EOSes that occur if no hypothesis is updated on several last steps
        actual_length = infer_length(last_stack.best_out, model.out_voc.eos)
        max_length = tf.reduce_max(actual_length)
        last_stack = last_stack._replace(best_out=last_stack.best_out[:, :max_length])

        self.best_out = last_stack.best_out
        self.best_attnP = last_stack.best_attnP
        self.best_scores = last_stack.best_scores
        self.best_raw_scores = last_stack.best_raw_scores
        self.best_state = last_stack.best_dec_state

    def translate_batch(self, batch_data, **optional_feed):
        """
        Translates NUMERIC batch of data
        :param batch_data: dict {'inp':np.array int32[batch,time]}
        :optional_feed: any additional values to be fed into graph. e.g. if you used placeholder for max_len at __init__
        :return: best hypotheses' outputs[batch, out_len] and attnP[batch, out_len, inp_len]
        """
        feed_dict = {placeholder: batch_data[k] for k, placeholder in self.batch_placeholder.items()}
        for k, v in optional_feed.items():
            feed_dict[k] = v

        out_ids, attnP = tf.get_default_session().run(
            [self.best_out, self.best_attnP],
            feed_dict=feed_dict)

        return out_ids, attnP

    def create_initial_stack(self, model, batch, **flags):
        """
        Creates initial stack for beam search by encoding inp and optionally forcing BOS as first output
        :type model: lib.task.seq2seq.inference.TranslateModel
        :param batch: model inputs - whatever model can eat for self.encode(batch,**tags)
        :param force_bos: if True, forces zero-th output to be model.out_voc.bos. Otherwise lets model decide.
        """

        dec_state = dec_state_0 = model.encode(batch, **flags)
        attnP_0 = model.get_attnP(dec_state_0)
        batch_size = tf.shape(attnP_0)[0]

        out_len = tf.constant(0, 'int32')
        out = tf.zeros(shape=(batch_size, 0), dtype=tf.int32)  # [batch_size, nout = 0]

        if self.force_bos:
            bos = tf.fill(value=model.out_voc.bos, dims=(batch_size,))
            dec_state = dec_state_1 = model.decode(dec_state_0, bos, **flags)
            attnP_1 = model.get_attnP(dec_state_1)
            attnP = tf.stack([attnP_0, attnP_1], axis=1)  # [batch_size, 2, ninp]
            out_len += 1
            out = tf.concat([out, bos[:, None]], axis=1)

        else:
            attnP = attnP_0[:, None, :]  # [batch_size, 1, ninp]

        slices = tf.range(0, batch_size)
        empty_out = tf.fill(value=model.out_voc.eos, dims=(batch_size, tf.shape(out)[1]))

        # Create stack.
        stack = self.Stack(
            out=out,
            scores=tf.zeros(shape=(batch_size,)),
            raw_scores=tf.zeros(shape=(batch_size,)),
            attnP=attnP,
            dec_state=dec_state,
            slices=slices,
            out_len=out_len,
            best_out=empty_out,
            best_scores=tf.fill(value=-float('inf'), dims=(batch_size,)),
            best_raw_scores=tf.fill(value=-float('inf'), dims=(batch_size,)),
            best_attnP=attnP,
            best_dec_state=dec_state,
            ext={}
        )

        return stack

    def should_continue_translating(self, model, stack):
        """
        Returns a bool vector for all hypotheses where True means hypo should be kept, 0 means it should be dropped.
        A hypothesis is dropped if it is either finished or pruned by beam_spread or by beam_size
        Note: this function assumes hypotheses for each input sample are sorted by scores(best first)!!!
        """

        # drop finished hypotheses
        should_keep = tf.logical_not(
            tf.reduce_any(tf.equal(stack.out, model.out_voc.eos), axis=-1))  # [batch_size x beam_size]

        n_hypos = tf.shape(stack.out)[0]
        batch_size = tf.shape(stack.best_out)[0]
        batch_indices = hypo_to_batch_index(n_hypos, stack.slices)

        # prune by length
        if self.max_len is not None:
            within_max_length = tf.less_equal(stack.out_len, self.max_len)

            # if we're given one max_len per each sentence, repeat it for each batch
            if not is_scalar(self.max_len):
                within_max_length = tf.gather(within_max_length, batch_indices)

            should_keep = tf.logical_and(
                should_keep,
                within_max_length,
            )

        # prune by beam spread
        if self.beam_spread is not None:
            best_scores_for_hypos = tf.gather(stack.best_scores, batch_indices)
            pruned_by_spread = tf.less(stack.scores + self.beam_spread, best_scores_for_hypos)
            should_keep = tf.logical_and(should_keep, tf.logical_not(pruned_by_spread))

        if self.beam_spread_raw:
            best_raw_scores_for_hypos = tf.gather(stack.best_raw_scores, batch_indices)
            pruned_by_raw_spread = tf.less(stack.raw_scores + self.beam_spread_raw, best_raw_scores_for_hypos)
            should_keep = tf.logical_and(should_keep,
                                         tf.logical_not(pruned_by_raw_spread))


        # pruning anything exceeding beam_size
        if self.beam_size is not None:
            # This code will use a toy example to explain itself: slices=[0,2,5,5,8], n_hypos=10, beam_size=2
            # should_keep = [1,1,1,0,1,1,1,1,0,1] (two hypotheses have been pruned/finished)

            # 1. compute index of each surviving hypothesis globally over full batch,  [0,1,2,3,3,4,5,6,7,7]
            survived_hypo_id = tf.cumsum(tf.cast(should_keep, 'int32'), exclusive=True)
            # 2. compute number of surviving hypotheses for each batch sample, [2,2,3,1]
            survived_hypos_per_input = tf.bincount(batch_indices, weights=tf.cast(should_keep, 'int32'),
                                                   minlength=batch_size, maxlength=batch_size)
            # 3. compute the equivalent of slices for hypotheses excluding pruned: [0,2,4,4,7]
            slices_exc_pruned = tf.cumsum(survived_hypos_per_input, exclusive=True)
            # 4. compute index of surviving hypothesis within one sample (for each sample)
            # index of input sentence in batch:       inp0  /inp_1\  /inp_2\, /inp_3\
            # index of hypothesis within input:      [0, 1, 0, 1, 1, 0, 1, 2, 0, 0, 1]
            # 'e' = pruned earlier, 'x' - pruned now:         'e'         'x'   'e'
            beam_index = survived_hypo_id - tf.gather(slices_exc_pruned, batch_indices)

            # 5. prune hypotheses with index exceeding beam_size
            pruned_by_beam_size = tf.greater_equal(beam_index, self.beam_size)
            should_keep = tf.logical_and(should_keep, tf.logical_not(pruned_by_beam_size))

        return should_keep

    def beam_search_step_expand_hypos(self, model, stack, **flags):
        """
        Performs one step of beam search decoding. Samples new hypothesis to stack.
        :type model: lib.task.seq2seq.inference.TranslateModel
        :type stack: BeamSearchDecoder.BeamSearchStack
        """

        # Prune
        #     - Against best completed hypo
        #     - Against best hypo in beam
        #     - EOS translations
        #     - Against beam size

        should_keep = self.should_continue_translating(model, stack)

        hypo_indices = tf.where(should_keep)[:, 0]
        stack = self.shuffle_beam(model, stack, hypo_indices)

        # Compute penalties, if any
        base_scores = self.compute_base_scores(model, stack, **flags)

        # Get top-beam_size new hypotheses for each input.
        # Note: we assume sample returns hypo_indices from highest score to lowest, therefore hypotheses
        # are automatically sorted by score within each slice.
        hypo_indices, words, delta_raw_scores = model.sample(stack.dec_state, base_scores, stack.slices,
                                                             self.beam_size, **flags
                                                             )

        # hypo_indices, words and delta_raw_scores may contain -1/-1/-inf triples for non-available hypotheses.
        # This can only happen if for some input there were 0 surviving hypotheses OR beam_size > n_hypos*vocab_size
        # In either case, we want to prune such hypotheses
        valid_indices = tf.where(tf.not_equal(tf.reshape(hypo_indices, [-1]), -1))[:, 0]
        hypo_indices = tf.gather(tf.reshape(hypo_indices, [-1]), valid_indices)
        words = tf.gather(tf.reshape(words, [-1]), valid_indices)
        delta_raw_scores = tf.gather(tf.reshape(delta_raw_scores, [-1]), valid_indices)

        stack = self.shuffle_beam(model, stack, hypo_indices)
        dec_state = model.decode(stack.dec_state, words, **flags)
        step_attnP = model.get_attnP(dec_state)
        # step_attnP shape: [batch_size * beam_size, ninp]

        # collect stats for the next step
        attnP = tf.concat([stack.attnP, step_attnP[:, None, :]], axis=1) # [batch * beam_size, nout, ninp]
        out = tf.concat([stack.out, words[..., None]], axis=-1)
        out_len = stack.out_len + 1

        raw_scores = stack.raw_scores + delta_raw_scores

        return stack._replace(
            out=out,
            raw_scores=raw_scores,
            attnP=attnP,
            out_len=out_len,
            dec_state=dec_state,
        )

    def beam_search_step_update_best(self, model, stack, maintain_best_state=False, **flags):
        """
        Performs one step of beam search decoding. Removes hypothesis from (beam_size ** 2) stack.
        :type model: lib.task.seq2seq.inference.TranslateModel
        :type stack: BeamSearchDecoder.BeamSearchStack
        """

        # Compute sample id for each hypo in stack
        n_hypos = tf.shape(stack.out)[0]
        batch_indices = hypo_to_batch_index(n_hypos, stack.slices)

        # Mark finished hypos
        finished = tf.equal(stack.out[:, -1], model.out_voc.eos)

        if self.min_len is not None:
            below_min_length = tf.less(stack.out_len, self.min_len)
            if not is_scalar(self.min_len):
                below_min_length = tf.gather(below_min_length, batch_indices)

            finished = tf.logical_and(finished, tf.logical_not(below_min_length))

        if self.if_no_eos == 'last':
            # No hypos finished with EOS, but len == max_len, consider unfinished hypos
            reached_max_length = tf.equal(stack.out_len, self.max_len)
            if not is_scalar(self.max_len):
                reached_max_length = tf.gather(reached_max_length, batch_indices)

            have_best_out = tf.reduce_any(tf.not_equal(stack.best_out, model.out_voc.eos), 1)
            no_finished_alternatives = tf.gather(tf.logical_not(have_best_out), batch_indices)
            allow_unfinished_hypo = tf.logical_and(reached_max_length, no_finished_alternatives)

            finished = tf.logical_or(finished, allow_unfinished_hypo)

        # select best finished hypo for each input in batch (if any)
        finished_scores = tf.where(finished, stack.scores, tf.fill(tf.shape(stack.scores), -float('inf')))
        best_scores, best_indices = sliced_argmax(finished_scores[:, None], stack.slices, 1)
        best_scores, best_indices = best_scores[:, 0], stack.slices + best_indices[:, 0]
        best_indices = tf.clip_by_value(best_indices, 0, tf.shape(stack.out)[0] - 1)

        stack_is_nonempty = tf.not_equal(tf.shape(stack.out)[0], 0)

        # take the better one of new best hypotheses or previously existing ones
        new_is_better = tf.greater(best_scores, stack.best_scores)
        best_scores = tf.where(new_is_better, best_scores, stack.best_scores)

        new_best_raw_scores = tf.cond(stack_is_nonempty,
                                      lambda:tf.gather(stack.raw_scores, best_indices),
                                      lambda:stack.best_raw_scores)

        best_raw_scores = tf.where(new_is_better, new_best_raw_scores, stack.best_raw_scores)


        batch_size = tf.shape(stack.best_out)[0]
        eos_pad = tf.fill(value=model.out_voc.eos, dims=(batch_size, 1))
        padded_best_out = tf.concat([stack.best_out, eos_pad], axis=1)
        new_out = tf.cond(stack_is_nonempty,
                          lambda: tf.gather(stack.out, best_indices),
                          lambda: tf.gather(padded_best_out, best_indices) # dummy out, best indices are zeros
                          )
        best_out = tf.where(new_is_better, new_out, padded_best_out)

        zero_attnP = tf.zeros_like(stack.best_attnP[:, :1, :])
        padded_best_attnP = tf.concat([stack.best_attnP, zero_attnP], axis=1)
        new_attnP = tf.cond(stack_is_nonempty,
                            lambda: tf.gather(stack.attnP, best_indices),
                            lambda: tf.gather(padded_best_attnP, best_indices), # dummy attnP, best indices are zeros
                            )
        best_attnP = tf.where(new_is_better, new_attnP, padded_best_attnP)

        # if better translation is reached, update it's state too
        best_dec_state = stack.best_dec_state
        if maintain_best_state:
            new_best_dec_state = model.shuffle(stack.dec_state, best_indices)
            best_dec_state = model.switch(new_is_better, new_best_dec_state, stack.best_dec_state)

        return stack._replace(
            best_out=best_out,
            best_scores=best_scores,
            best_attnP=best_attnP,
            best_raw_scores=best_raw_scores,
            best_dec_state=best_dec_state,
        )

    def beam_search_step(self, model, stack, **flags):
        stack = self.beam_search_step_expand_hypos(model, stack, **flags)
        stack = stack._replace(
            scores=self.compute_scores(model, stack, **flags)
        )
        is_beam_not_empty = tf.not_equal(tf.shape(stack.raw_scores)[0], 0)
        return self.beam_search_step_update_best(model, stack, **flags)

    def compute_scores(self, model, stack, **flags):
        """
        Compute hypothesis scores given beam search stack. Applies any penalties necessary.
        For quick prototyping, you can store whatever penalties you need in stack.dec_state
        :type model: lib.task.seq2seq.inference.TranslateModel
        :type stack: BeamSearchDecoder.BeamSearchStack
        :return: float32 vector (one score per hypo)
        """
        return stack.raw_scores

    def compute_base_scores(self, model, stack, **flags):
        """
        Compute hypothesis scores to be used as base_scores for model.sample.
        This is usually same as compute_scores but scaled to the magnitude of log-probabilities
        :type model: lib.task.seq2seq.inference.TranslateModel
        :type stack: BeamSearchDecoder.BeamSearchStack
        :return: float32 vector (one score per hypo)
        """
        return self.compute_scores(model, stack, **flags)

    def shuffle_beam(self, model, stack, flat_indices):
        """
        Selects hypotheses by index from entire BeamSearchStack
        Note: this method assumes that both stack and flat_indices are sorted by sample index
        (i.e. first are indices for input0 are, then indices for input1, then 2, ... then input[batch_size-1]
        """
        n_hypos = tf.shape(stack.out)[0]
        batch_size = tf.shape(stack.best_out)[0]

        # compute new slices:
        # step 1: get index of inptut sequence (in batch) for each hypothesis in flat_indices
        sample_ids_for_slices = tf.gather(hypo_to_batch_index(n_hypos, stack.slices), flat_indices)
        # step 2: compute how many hypos per flat_indices
        n_hypos_per_sample = tf.bincount(sample_ids_for_slices, minlength=batch_size, maxlength=batch_size)
        # step 3: infer slice start indices
        new_slices = tf.cumsum(n_hypos_per_sample, exclusive=True)

        # shuffle everything else
        return stack._replace(
            out=tf.gather(stack.out, flat_indices),
            scores=tf.gather(stack.scores, flat_indices),
            raw_scores=tf.gather(stack.raw_scores, flat_indices),
            attnP=tf.gather(stack.attnP, flat_indices),
            dec_state=model.shuffle(stack.dec_state, flat_indices),
            ext=nested_map(lambda x: tf.gather(x, flat_indices), stack.ext),
            slices=new_slices,
        )


class PenalizedBeamSearchDecoder(BeamSearchDecoder):
    """
    Performs ingraph beam search for given input sequences (inp)
    Implements length and coverage penalties
    """
    PenalizedExt = namedtuple('PenalizedExt', [
        'attnP_sum',  # [batch_size x beam_size, ninp]
    ])

    def beam_search_step_expand_hypos(self, model, stack, **flags):
        new_stack = super().beam_search_step_expand_hypos(model, stack, **flags)
        new_stack_ext = new_stack.ext[self.PenalizedExt]

        step_attnP = model.get_attnP(new_stack.dec_state)
        # step_attnP shape: [batch_size * beam_size, ninp]

        new_stack.ext[self.PenalizedExt] = new_stack_ext._replace(
            attnP_sum=new_stack_ext.attnP_sum + step_attnP)
        return new_stack

    def create_initial_stack(self, model, batch, **flags):
        stack = super().create_initial_stack(model, batch, **flags)
        stack.ext[self.PenalizedExt] = self.PenalizedExt(
            attnP_sum=tf.reduce_sum(stack.attnP, axis=1))
        return stack

    def compute_scores(self, model, stack, len_alpha=1, attn_beta=0, **flags):
        """
        Computes scores after length and coverage penalty
        :param len_alpha: coefficient for length penalty, score / ( [5 + len(output_sequence)] / 6) ^ len_alpha
        :param attn_beta: coefficient for coverage penalty (additive)
            attn_beta * sum_i {log min(1.0, sum_j {attention_p[x_i,y_j] }  )}
        :return: float32 vector (one score per hypo)
        """
        stack_ext = stack.ext[self.PenalizedExt]

        if attn_beta:
            warn("whenever attn_beta !=0, this code works as in http://bit.ly/2ziK5a8,"
                 "which may or may not be correct depending on your definition.")

        scores = stack.raw_scores
        if len_alpha:
            length_penalty = tf.pow((1. + tf.to_float(stack.out_len) / 6.), len_alpha)
            scores /= length_penalty
        if attn_beta:
            times_translated = tf.minimum(stack_ext.attnP_sum, 1)
            coverage_penalty = tf.reduce_sum(
                tf.log(times_translated + sys.float_info.epsilon),
                axis=-1) * attn_beta
            scores += coverage_penalty
        return scores

    def compute_base_scores(self, model, stack, len_alpha=1, **flags):
        """
        Compute hypothesis scores to be used as base_scores for model.sample
        :return: float32 vector (one score per hypo)
        """
        scores = self.compute_scores(model, stack, len_alpha=len_alpha, **flags)
        if len_alpha:
            length_penalty = tf.pow((1. + tf.to_float(stack.out_len) / 6.), len_alpha)
            scores *= length_penalty
        return scores


def get_words_attnP(step_attnP, inp_words_mask, slices, src_word_attn_aggregation='max'):
    # Helper function to extract word-level alignment aggregation on src.
    # For parameter description see AlignmentPenaltyBeamSearchDecoder.AlignmentPenaltyExt

    def _get_words_attnP(step_attnP, inp_words_mask, slices):
        max_words_len = np.max(np.sum(inp_words_mask, axis=1))
        words_attnP = np.zeros((step_attnP.shape[0], max_words_len))
        slices = slices.tolist() + [step_attnP.shape[0]]
        for words_mask, (b, e) in zip(inp_words_mask,
                                      zip(slices[:-1], slices[1:])):
            words_ind = np.where(words_mask)[0].tolist() + [len(words_mask)]
            for i, (wb, we) in enumerate(zip(words_ind[:-1], words_ind[1:])):
                if src_word_attn_aggregation == 'max':
                    words_attnP[b:e, i] = np.max(step_attnP[b:e, wb:we], axis=1)
                elif src_word_attn_aggregation == 'sum':
                    words_attnP[b:e, i] = np.sum(step_attnP[b:e, wb:we], axis=1)
                else:
                    raise ValueError('Unknown src_word_attn_aggregation mode: %s' % src_word_attn_aggregation)
        return words_attnP.astype(np.float32)

    words_attnP = tf.py_func(_get_words_attnP, [step_attnP, inp_words_mask, slices], tf.float32, stateful=False)
    words_attnP.set_shape([None, None])
    return tf.stop_gradient(words_attnP)


class AlignmentPenaltyBeamSearchDecoder(BeamSearchDecoder):
    AlignmentPenaltyExt = namedtuple('AlignmentPenaltyExt', [
        'attnP_aggregated_src',  # [batch_size x beam_size, ninp|ninp_words]
        'attnP_aggregated_dst',  # [batch_size x beam_size, nout]

        'inp_words_mask', # Does bpe token start a new word? [batch_size, ninp], bool
    ])

    def __init__(self, *args,
                 len_alpha=1,
                 attn_beta=0, src_attn_aggregation='max',
                 src_word_attn_aggregation=None,
                 dst_attn_beta=0, dst_attn_aggregation='max',
                 **kwargs ):
        # We need to initialize them all to create initial stack
        self.len_alpha = len_alpha
        self.attn_beta = attn_beta
        self.src_attn_aggregation = src_attn_aggregation
        self.src_word_attn_aggregation = src_word_attn_aggregation
        self.dst_attn_beta = dst_attn_beta
        self.dst_attn_aggregation = dst_attn_aggregation
        super().__init__(*args, **kwargs)

    def beam_search_step_expand_hypos(self, model, stack, **flags):
        stack = super().beam_search_step_expand_hypos(model, stack, **flags)
        stack_ext = stack.ext[self.AlignmentPenaltyExt]

        step_attnP = model.get_attnP(stack.dec_state)
        # step_attnP shape: [batch_size * beam_size, ninp]

        # updating attnP_aggregated_src
        step_attnP_word = step_attnP
        if self.src_word_attn_aggregation:
            step_attnP_word = get_words_attnP(
                step_attnP_word, stack_ext.inp_words_mask,
                stack.slices, self.src_word_attn_aggregation)

        max_words_num = tf.shape(stack_ext.attnP_aggregated_src)[1]
        paddings = max_words_num - tf.shape(step_attnP_word)[1]
        step_attnP_word = tf.pad(step_attnP_word, [[0, 0], [0, paddings]])

        if self.attn_beta:
            if self.src_attn_aggregation == 'max':
                attnP_aggregated_src = tf.maximum(stack_ext.attnP_aggregated_src,
                                                  step_attnP_word)
            elif self.src_attn_aggregation == 'sum':
                attnP_aggregated_src = stack_ext.attnP_aggregated_src + step_attnP_word
            else:
                raise ValueError
        else:
            attnP_aggregated_src = stack_ext.attnP_aggregated_src

        # updating attnP_aggregated_dst
        if self.dst_attn_beta:
            if self.dst_attn_aggregation == 'max':
                dst_attnP_aggregated = tf.reduce_max(step_attnP_word, axis=-1)[:, None]
            elif self.dst_attn_aggregation == 'sum':
                dst_attnP_aggregated = tf.reduce_sum(step_attnP_word, axis=-1)[:, None]
            else:
                raise ValueError

            attnP_aggregated_dst = tf.concat(
                [stack_ext.attnP_aggregated_dst, dst_attnP_aggregated],
                axis=1)
        else:
            attnP_aggregated_dst = stack_ext.attnP_aggregated_dst

        stack.ext[self.AlignmentPenaltyExt] = stack_ext._replace(
            attnP_aggregated_src=attnP_aggregated_src,
            attnP_aggregated_dst=attnP_aggregated_dst)
        return stack

    def create_initial_stack(self, model, batch, **flags):
        stack = super().create_initial_stack(model, batch, **flags)

        words_attnP = tf.squeeze(stack.attnP, axis=1)

        # Calc inp_words_mask and aggregate data.
        if self.src_word_attn_aggregation:
            def is_new_word(inp_words):
                return np.array([[not v.startswith(b'`') for v in l] for l in inp_words])

            inp_words_mask = tf.py_func(is_new_word, [batch['inp_words']], bool, stateful=False)
            inp_words_mask.set_shape(batch['inp_words'].shape)
            inp_words_mask = tf.stop_gradient(inp_words_mask)

            words_attnP = get_words_attnP(
                words_attnP, inp_words_mask, stack.slices,
                self.src_word_attn_aggregation)
        else:
            inp_words_mask = tf.fill(tf.shape(batch['inp']), 1.0)


        if self.attn_beta:
            if self.src_attn_aggregation in ('max', 'sum'):
                attnP_aggregated_src = words_attnP
            else:
                raise ValueError
        else:
            attnP_aggregated_src = tf.fill(tf.shape(batch['inp']), 0.0)

        # Calc attnP_aggregated_dst
        if self.dst_attn_beta:
            if self.dst_attn_aggregation == 'max':
                attnP_aggregated_dst = tf.reduce_max(words_attnP, axis=-1)
            elif self.dst_attn_aggregation == 'sum':
                attnP_aggregated_dst = tf.reduce_sum(words_attnP, axis=-1)
            else:
                raise ValueError
        else:
            attnP_aggregated_dst = tf.fill((tf.shape(batch['inp'])[0],), 0.0)
        attnP_aggregated_dst = attnP_aggregated_dst[:, None]

        stack.ext[self.AlignmentPenaltyExt] = self.AlignmentPenaltyExt(
            attnP_aggregated_src=attnP_aggregated_src,
            attnP_aggregated_dst=attnP_aggregated_dst,
            inp_words_mask=inp_words_mask
        )
        return stack

    def compute_scores(self, model, stack, **flags):
        """
        Computes scores after length and coverage penalty
        :param len_alpha: coefficient for length penalty, score / ( [5 + len(output_sequence)] / 6) ^ len_alpha
        :param attn_beta: coefficient for coverage penalty (additive)
            attn_beta * sum_i {log min(1.0, {src_attn_aggregation}_j {attention_p[x_i,y_j] }  )}
        :param src_attn_aggregation: aggregation for src coverage penalty.
            Possible values are 'max', 'sum'.
        :param src_word_attn_aggregation: should we aggregate src coverage penalty by words?
            Possible values are None/max/sum.
        :param dst_attn_beta: coefficient for coverage penalty on dst side:
            attn_beta * sum_j {log min(1.0, {dst_attn_aggregation}_i {attention_p[x_i,y_j] }  )}
        :param dst_attn_aggregation: aggregation for dst coverage penalty.
            Possible values are 'max', 'sum'.
        :return: float32 vector (one score per hypo)
        """

        stack_ext = stack.ext[self.AlignmentPenaltyExt]

        scores = stack.raw_scores
        if self.len_alpha:
            length_penalty = tf.pow((1. + tf.to_float(stack.out_len) / 6.), self.len_alpha)
            scores /= length_penalty
        if self.attn_beta:
            coverage_penalty = tf.reduce_sum(
                tf.log(tf.minimum(stack_ext.attnP_aggregated_src, 1) + sys.float_info.epsilon),
                axis=-1)
            scores += coverage_penalty * self.attn_beta
        if self.dst_attn_beta:
            coverage_penalty = tf.reduce_sum(
                tf.log(tf.minimum(stack_ext.attnP_aggregated_dst, 1) + sys.float_info.epsilon),
                axis=-1)
            scores += coverage_penalty * self.dst_attn_beta

        return scores

    def compute_base_scores(self, model, stack, **flags):
        """
        Compute hypothesis scores to be used as base_scores for model.sample
        :return: float32 vector (one score per hypo)
        """
        scores = self.compute_scores(model, stack, **flags)
        if self.len_alpha:
            length_penalty = tf.pow((1. + tf.to_float(stack.out_len) / 6.), self.len_alpha)
            scores *= length_penalty
        return scores


def hypo_to_batch_index(n_hypos, slices):
    """
    Computes index in batch (input sequence index) for each hypothesis given slices.
    :param n_hypos: number of hypotheses (tf int scalar)
    :param slices: indices of first hypo for each input in batch

    It should guaranteed that
     - slices[0]==0 (first hypothesis starts at index 0), otherwise output[:slices[0]] will be -1
     - if batch[i] is terminated, then batch[i]==batch[i+1]
    """
    is_next_sent_at_t = tf.bincount(slices, minlength=n_hypos, maxlength=n_hypos)
    hypo_to_index = tf.cumsum(is_next_sent_at_t) - 1
    return hypo_to_index
