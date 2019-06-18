#!/usr/bin/env python3

# This is a hideous collection of scripts for machine translation tasks for training, translating, etc.
# If you don't want a Ph.D in configuring this file, simply jump to main.py and continue from there

import argparse
import json
import math
import numpy as np
import tensorflow as tf
import time
import operator
import os
import sys
import pickle
import glob
import itertools


from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2

# add local libs to pythonpath
sys.path.insert(0, os.path.dirname(__file__) + '/..')

import lib
from lib.data import CostBufferIterator, RoundRobinIterator, ShuffleIterator
from lib.train.tickers import *
from lib.task.seq2seq.tickers import *
from lib.task.seq2seq.problems.default import DefaultProblem
from lib.task.seq2seq.data import load_parallel, random_block_reader, filter_by_len, locally_sorted_by_len, maxlen, \
    batch_cost, form_batches, form_adaptive_batches, form_adaptive_batches_windowed, form_adaptive_batches_split2d

from lib.session import ProfilableSessionWrapper, profile_scope, get_profile_level, set_profile_level
from lib.util import load_class, merge_dicts

from tensorflow.python import debug as tf_debug

np.set_printoptions(threshold=300000000, linewidth=1000, suppress=True)
SEED = 'zemberek'  # the secret holy grail of Yandex Translate


def rename_nodes(graph_def, node_name_dict):
    output_graph_def = graph_pb2.GraphDef()
    for node in graph_def.node:
        output_node = node_def_pb2.NodeDef()
        output_node.CopyFrom(node)

        output_node.name = node_name_dict.get(node.name, node.name)

        output_node.ClearField("input")
        for node_name in node.input:
            output_node.input.extend([node_name_dict.get(node_name, node_name)])

        output_graph_def.node.extend([output_node])
    return output_graph_def


def get_node_name(op):
    return re.sub("\:\d+$", "", op.name)


def load_default_problem(model_class_name, inp_voc_path, out_voc_path, hp, problem_config={}, max_srclen=None,
                         align=None, feat_vocs=None):
    # Load vocs.
    with open(inp_voc_path, 'rb') as f:
        inp_voc = pickle.load(f)
    with open(out_voc_path, 'rb') as f:
        out_voc = pickle.load(f)
    feat_vocs = []
    for feat_voc in feat_vocs:
        with open(feat_voc, 'rb') as f:
            feat_vocs.append(pickle.load(f))

    if os.path.exists(model_class_name):
        # Read Python file to get model class
        with open(model_class_name, 'rt') as f:
            text = f.read()
        code = compile(text, model_class_name, 'exec')
        vars = {}
        exec(code, vars, vars)
        model_cls = vars['Model']
    else:
        model_cls = load_class(model_class_name)

    # Create model instance
    if align:
        with open(align, 'rb') as f:
            align = pickle.load(f)
        hp['align'] = align
    if len(feat_vocs) > 0:
        hp['feat_vocs'] = feat_vocs
    return DefaultProblem({'mod': model_cls('mod', inp_voc, out_voc, **hp)}, **problem_config)


def load_problem(args, **kwargs):
    model_specs = getattr(args, 'models', None)
    if model_specs is None or len(model_specs) == 0:
        model_specs = args.models = {'mod': {'class': args.model, 'translation_mode': 'src2dst', 'dev_suffix': ''}}

    if not hasattr(args, 'problem') or args.problem is None:
        return load_default_problem(args.model, args.ivoc, args.ovoc, args.hp, **kwargs)

    problem_class = load_class(args.problem)

    model_spec_options = {'class', 'translation_mode', 'device', 'hp', 'inp_voc', 'out_voc', 'dev_suffix'}

    models = {}
    for model_name, model_spec in model_specs.items():
        assert len(set(model_spec.keys()).intersection(model_spec_options)) == len(model_spec.keys()), Exception(
            'invalid model options')

        model_class = load_class(model_spec['class'])

        model_translation_mode = model_spec.get('translation_mode')
        assert model_translation_mode in [None, 'src2dst', 'dst2src'], Exception("invalid translation_mode")

        model_device = model_spec.get('device')
        model_inp_voc_path = model_spec.get('inp_voc', args.ivoc if model_translation_mode != 'dst2src' else args.ovoc)
        model_out_voc_path = model_spec.get('out_voc', args.ovoc if model_translation_mode != 'dst2src' else args.ivoc)

        model_hp = merge_dicts(args.hp, model_spec.get('hp', {}))

        with open(model_inp_voc_path, 'rb') as f:
            model_inp_voc = pickle.load(f)

        with open(model_out_voc_path, 'rb') as f:
            model_out_voc = pickle.load(f)
        with tf.device(model_device) if model_device else lib.util.nop_ctx():
            models[model_name] = model_class(name=model_name, inp_voc=model_inp_voc, out_voc=model_out_voc, **model_hp)

    return problem_class(models, **args.problem_opts)


def set_no_constraints(hp):
    hp['is_constrained'] = False


def make_random_vec(size):
    CLIP = math.sqrt(3.0 / size)
    return np.array([np.random.uniform(-CLIP, CLIP)
                     for _ in range(size)], dtype=np.float32)


def average_checkpoints(folder, num_checkpoints):
    model_vars = {}
    checkpoints = []
    for fname in os.listdir(folder):
        if not fname.startswith('model-') or not fname.endswith('.npz'):
            continue
        label = fname[len('model-'):-len('.npz')]
        if not label.isdigit():
            continue
        checkpoints.append(int(label))
    checkpoints = sorted(checkpoints, reverse=True)[0:num_checkpoints]
    for checkpoint in checkpoints:
        filename = os.path.join(folder, 'model-%d.npz' % checkpoint)
        checkpoint = np.load(filename)
        for var in checkpoint:
            if var in model_vars:
                model_vars[var] += checkpoint[var]
            else:
                model_vars[var] = checkpoint[var]
    for var in model_vars:
        model_vars[var] /= len(checkpoints)
    np.savez(os.path.join(folder, 'model-latest.npz'), **model_vars)


## ============================================================================
#                                  'mkvoc'

def MKVOC_add_params(subp):
    p = subp.add_parser('mkvoc')
    p.add_argument('--text', required=True)
    p.add_argument('--outvoc', required=True)
    p.add_argument('--n-words', required=True, type=int)
    p.add_argument('--unk-num', action='store_true', default=False)
    p.add_argument('--index', type=int, default=0)
    return 'mkvoc'


def MKVOC(args):
    voc = lib.task.seq2seq.voc.Voc.compile(args.text, args.n_words, args.index)
    with open(args.outvoc, 'wb') as f:
        pickle.dump(voc, f)


## ============================================================================
#                                 'train'

def TRAIN_add_params(subp):
    eval_arg = lambda x: eval(x, locals(), globals())

    p = subp.add_parser('train')
    p.add_argument('--model', required=False)
    p.add_argument('--ivoc', required=False)
    p.add_argument('--ovoc', required=False)
    p.add_argument('--max-srclen', required=True, type=int)
    p.add_argument('--max-dstlen', required=True, type=int)
    p.add_argument('--batch-maker', default='adaptive_windowed')
    p.add_argument('--maxlen-min', default=1, type=int)
    p.add_argument('--maxlen-quant', default=1, type=int)
    p.add_argument('--split-alterate', default=False, action='store_true')
    p.add_argument('--t2t-batches', type=str)
    p.add_argument('--t2t-batches-start', type=int, default=0)
    p.add_argument('--train-src', required=True)
    p.add_argument('--train-dst', required=True)
    p.add_argument('--train-paste', action='store_true', default=False, help='SRC/DST separated by <tab>')
    p.add_argument('--dev-src', required=True, action='append')
    p.add_argument('--dev-dst', required=True, action='append')
    p.add_argument('--num-batches', type=int, default=10000000)
    p.add_argument('--part-size', type=int, default=1024 * 1024)
    p.add_argument('--part-parallel', type=int, default=16)
    p.add_argument('--batch-len', type=int, default=5000)
    p.add_argument('--shuffle-len', type=int, default=12000)
    p.add_argument('--batch-shuffle-len', type=int, default=0)
    p.add_argument('--batch-size-max', type=int, default=0)  # 0 - no batch size limit
    p.add_argument('--cost-buffer-len', type=int, default=12000)
    p.add_argument('--split-len', type=int, default=10000)
    p.add_argument('--split-chunk-count', type=int, default=0)
    p.add_argument('--lead-inp-len', default=False, action='store_true')
    p.add_argument('--folder', required=True)
    p.add_argument('--device', type=str, default='')
    p.add_argument('--gpu-memory-fraction', type=float, default=.95)
    p.add_argument('--gpu-allow-growth', action='store_true', default=False)
    p.add_argument('--align', type=str, default='')
    p.add_argument('--decay-after-batches', default=0, type=int)
    p.add_argument('--profile', action='store_true', default=False)
    p.add_argument('--profile-level', type=int, default=1, help='Profiling verbosity level')
    p.add_argument('--tfdbg', action='store_true', default=False)
    p.add_argument('--translate-dev', action='store_true', default=False)
    p.add_argument('--translate-dev-every', default=4096, type=int)
    p.add_argument('--translate-dev-initial', action='store_true', default=False)
    p.add_argument('--score-dev-every', default=256, type=int)
    p.add_argument('--score-dev-initial', action='store_true', default=False)
    p.add_argument('--learning-rate-stop-value', default=0.0, type=float)
    p.add_argument('--rollback-off', action='store_true', default=False)
    p.add_argument('--feat-vocs', action='append')
    p.add_argument('--average-checkpoints', type=int, default=1)
    p.add_argument('--checkpoint-every-steps', type=int, default=2048)
    p.add_argument('--checkpoint-every-minutes', type=int, default=0)
    p.add_argument('--time-every-steps', type=int)
    p.add_argument('--learning-rate', type=float, default=.001)
    p.add_argument('--decay-policy', default='constant')
    p.add_argument('--decay-steps', type=int)
    p.add_argument('--decay-min-steps', type=int, default=0)
    p.add_argument('--optimizer', type=str, default='adam')
    p.add_argument('--optimizer-opts', type=eval_arg, default={})
    p.add_argument('--no-skip', dest='skip_train_data', default=True, action='store_false')
    p.add_argument('--reader-seed', type=int, default=42)
    p.add_argument('--dump-dir', default=None)
    p.add_argument('--dump-first-n', default=None, type=int)
    p.add_argument('--keep-checkpoints-max', type=int)
    p.add_argument('--hp', type=eval_arg, default={})
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--pre-init-model-checkpoint', type=str)
    p.add_argument('--pre-init-state-checkpoint', type=str)
    p.add_argument('--problem', type=str)
    p.add_argument('--problem-opts', type=eval_arg, default={})
    p.add_argument('--models', type=eval_arg, default={})
    p.add_argument('--mpi-provider', type=str)
    p.add_argument('--mpi-gpu-count-per-process', type=int, default=1)
    p.add_argument('--summary-every-steps', type=int, default=1)
    p.add_argument('--params-summary-every-steps', type=int, default=2048)
    p.add_argument('--avg_checkpoint_every', type=int, default=None)
    p.add_argument('--avg_last_checkpoints', type=int, default=None)
    p.add_argument('--translate-on-master', dest='translate_parallel', action='store_false')
    p.add_argument('--end-of-params', action='store_true', default=False)
    return 'train'


def TRAIN(args):
    if not args.end_of_params:
        raise ValueError(
            "You have forgotten to pass --end-of-params. This probably means that there is an extra space in your train script and not all parameters are present.")

    # Set random seed.
    tf.set_random_seed(args.seed)

    # multi-gpu: we need Local Rank since it is the number of GPU on this host
    MPI_RANK = os.getenv('OMPI_COMM_WORLD_RANK')
    MPI_LOCAL_RANK = os.getenv('OMPI_COMM_WORLD_LOCAL_RANK')
    MPI_SIZE = os.getenv('OMPI_COMM_WORLD_SIZE')

    # Load problem and corpora.
    problem = load_problem(
        args,
        problem_config=args.problem_opts,
        max_srclen=args.max_srclen,
        align=args.align,
        feat_vocs=args.feat_vocs)

    # Set mpi provider from args if given
    if args.mpi_provider is not None:
        lib.ops.mpi.set_provider(args.mpi_provider)

    # Form batches and epoch infinite generator
    def batch_form_func(x):
        rng = np.random.RandomState(42)

        def weight_func(x, min_maxlen=args.maxlen_min, quant=args.maxlen_quant):
            out = maxlen(x)
            out = math.ceil(out / quant) * quant
            out = max(min_maxlen, out)
            return out

        if args.batch_maker == 'adaptive_windowed':
            x = form_adaptive_batches_windowed(
                x,
                weight_func=weight_func,
                max_size=args.batch_len,
                split_len=args.split_len,
                batch_size_max=args.batch_size_max)
        elif args.batch_maker == 'adaptive':
            x = locally_sorted_by_len(x, args.split_len, weight_func=weight_func, alterate=args.split_alterate)
            x = form_adaptive_batches(x, args.batch_len, batch_size_max=args.batch_size_max)
            x = ShuffleIterator(x, args.batch_shuffle_len)
        elif args.batch_maker == 'single_example':
            x = ([xi] for xi in x)
        elif args.batch_maker == 'simple':
            x = form_batches(x, args.batch_size)
        else:
            raise Exception("Unexpected batch_maker:", args.batch_maker)
        return x

    learning_rate_fn = LearningRateFn(
        policy=args.decay_policy, scale=args.learning_rate,
        decay_steps=args.decay_steps, hid_size=args.hp['hid_size']
    )

    if args.optimizer == 'lazy_adam':
        algo = lib.train.algorithms.LazyAdam(
            learning_rate=learning_rate_fn,
            **args.optimizer_opts)
    elif args.optimizer == 'adam':
        algo = lib.train.algorithms.Adam(
            learning_rate=learning_rate_fn,
            **args.optimizer_opts)
    elif args.optimizer == 'sgd':
        algo = lib.train.algorithms.Sgd(
            learning_rate=learning_rate_fn,
            **args.optimizer_opts)
    elif args.optimizer == 'rms_prop':
        algo = lib.train.algorithms.RMSProp(
            learning_rate=learning_rate_fn,
            **args.optimizer_opts)
    else:
        raise Exception('Unsupported optimizer %s' % args.optimizer)

    # Create session
    config = tf.ConfigProto()
    config.gpu_options.visible_device_list = args.device.replace('/gpu:', '') if MPI_RANK is None else MPI_LOCAL_RANK
    config.gpu_options.per_process_gpu_memory_fraction = args.gpu_memory_fraction
    config.allow_soft_placement = True

    if args.gpu_allow_growth:
        config.gpu_options.allow_growth = True

    if MPI_LOCAL_RANK is not None:
        session = lib.ops.mpi.Session(config=config, gpu_group=int(MPI_LOCAL_RANK),
                                      gpu_group_size=args.mpi_gpu_count_per_process)
    else:
        session = tf.Session(config=config)

    if args.profile:
        session = ProfilableSessionWrapper(session, log_dir=os.path.join(args.folder, 'train_log'),
                                           profile_level=args.profile_level)

    if args.tfdbg:
        session = tf_debug.LocalCLIDebugWrapperSession(session)

    save_folder = os.path.join(args.folder, 'checkpoint')
    summary_folder = os.path.join(args.folder, 'summary')
    translations_folder = os.path.join(args.folder, 'translations')

    if lib.ops.mpi.is_master():
        if args.train_paste:
            train = random_block_reader(args.train_src, part_size=args.part_size, parallel=args.part_parallel,
                                        infinite=True, seed=args.reader_seed)
            train = (line.split('\t', 2)[:2] for line in train)
        else:
            train = load_parallel(args.train_src, args.train_dst, cycle=True)

        train = filter_by_len(train, args.max_srclen, args.max_dstlen, args.batch_len)
        train = ShuffleIterator(train, args.shuffle_len)
        train = batch_form_func(train)
        train = ((x, batch_cost(x)) for x in train)

        devs = []
        for dev_src, dev_dst in zip(args.dev_src, args.dev_dst):
            dev = load_parallel(dev_src, dev_dst)
            dev = filter_by_len(dev, args.max_srclen, args.max_dstlen, args.batch_len)
            dev = batch_form_func(dev)
            devs.append(dev)

        os.makedirs(save_folder, exist_ok=True)
        os.makedirs(summary_folder, exist_ok=True)

        full_devs = [list(dev) for dev in devs]  # full dev sets even if there are several nodes
        devs = [iter(dev) for dev in full_devs]
    else:
        devs = [None] * len(args.dev_src)
        full_devs = [[] for _ in range(len(args.dev_src))]
        train = None

    train = RoundRobinIterator(train, with_cost=args.cost_buffer_len > 0)
    if args.cost_buffer_len > 0:
        train = CostBufferIterator(train, args.cost_buffer_len)
    devs = [RoundRobinIterator(dev if lib.ops.mpi.is_master() else None, is_train=False) for dev in devs]

    # iterator over train feed dicts
    train_np = map(lambda b: problem.make_feed_dict(b, is_train=True), train)

    tickers = [
        SaveLoad(save_folder,
                 every_steps=args.checkpoint_every_steps, every_minutes=args.checkpoint_every_minutes,
                 skip_train_data=args.skip_train_data, keep_checkpoints_max=args.keep_checkpoints_max,
                 pre_init_model_checkpoint=args.pre_init_model_checkpoint,
                 pre_init_state_checkpoint=args.pre_init_state_checkpoint,
                 avg_checkpoint_every=args.avg_checkpoint_every,
                 avg_last_checkpoints=args.avg_last_checkpoints
                 ),
        Summary(summary_folder, params_every_steps=args.params_summary_every_steps,
                every_steps=args.summary_every_steps),
        GlobalStepStopper(args.num_batches),
    ]

    if args.time_every_steps is not None:
        tickers.append(TimeTicker(every_steps=args.time_every_steps))

    if args.decay_after_batches is not None and args.decay_after_batches > 0:
        tickers += [
            DecayLearningRate(after_steps=args.decay_after_batches, rollback=not args.rollback_off),
        ]
    tickers += [
        LearningRateStopper(
            threshold=args.learning_rate_stop_value,
            min_steps=args.decay_min_steps),
    ]

    with session:
        for i, (dev, full_dev) in enumerate(zip(devs, full_devs)):
            name = 'Dev' if i == 0 else 'Dev{}'.format(i)
            dev = list(dev)
            backward_dev = [[[row[1], row[0]] for row in batch] for batch in dev]
            backward_full_dev = [[[row[1], row[0]] for row in batch] for batch in full_dev]
            dev_np = map(problem.make_feed_dict, dev)
            tickers.append(
                DevLossTicker(dev_np, name=name, every_steps=args.score_dev_every, initial=args.score_dev_initial))

            if args.translate_dev:
                for model_name, model in sorted(args.models.items()):
                    translation_mode = model.get('translation_mode')
                    assert translation_mode in ['src2dst', 'dst2src', None]

                    if translation_mode is not None:
                        # if suffix is not passed, TranslateTicker will add model_name as a suffix

                        if args.translate_parallel:
                            translate_data = dev if translation_mode != 'dst2src' else backward_dev
                        else:
                            # if translation only happens on master, give it full data
                            translate_data = full_dev if translation_mode != 'dst2src' else backward_full_dev

                tickers.append(TranslateTicker(model_name, translate_data,
                                               name=name, folder=translations_folder,
                                               every_steps=args.translate_dev_every,
                                               initial=args.translate_dev_initial,
                                               suffix=model.get('dev_suffix'),
                                               device=model.get('device'),
                                               parallel=args.translate_parallel, ), )

        lib.train.train(problem, algo, train_np, tickers,
                        tick_every_steps=args.optimizer_opts.get('sync_every_steps', 0))

    if args.average_checkpoints > 1:
        if lib.ops.mpi.is_master():
            average_checkpoints(save_folder, args.average_checkpoints)



## ============================================================================
#                                   Main

def main():
    # Create parser.
    p = argparse.ArgumentParser('nmt.py')
    subp = p.add_subparsers(dest='cmd')

    # Add subcommands.
    mkvoc = MKVOC_add_params(subp)
    train = TRAIN_add_params(subp)

    # Parse.
    args = p.parse_args()

    # Run.
    if args.cmd == mkvoc:
        MKVOC(args)
    elif args.cmd == train:
        TRAIN(args)
    else:
        p.print_help()


if __name__ == '__main__':
    main()

