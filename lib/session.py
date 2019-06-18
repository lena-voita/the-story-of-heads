import tensorflow as tf
from tensorflow.python import ops
import lib
import sys
import os
import threading
from contextlib import contextmanager
from tensorflow.python.framework import *
from tensorflow.contrib.tfprof import *
from tensorflow.python.client import timeline, session
from collections import namedtuple

# tfprof-oriented Session object.
# More about tfprof:
# https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/tfprof
#
# !! Attention !! For using need to append
# /usr/local/cuda-8.0/extras/CUPTI/lib64 to $LD_LIBRARY_PATH


PROFILE_SUPER_VERBOSE = 666

_tls = threading.local()
def get_profile_level():
    if not hasattr(_tls, 'profile_level'):
        _tls.profile_level = PROFILE_SUPER_VERBOSE  # Never profile most of sess.run
    return _tls.profile_level

def set_profile_level(level):
    _tls.profile_level = level

@contextmanager
def profile_scope(level=1):
    prev_level = get_profile_level()
    _tls.profile_level = level
    try:
        yield
    finally:
        _tls.profile_level = prev_level


MemTimelineRecord = namedtuple('MemTimelineRecord', ['ts', 'node_name', 'bytes_in_use', 'live_bytes'])


class SessionWrapper(session.SessionInterface):

    def __init__(self, session):
        self._sess = session

    @property
    def graph(self):
        return self._sess.graph

    @property
    def sess_str(self):
        return self._sess.sess_str

    def run(self, *a, **kwa):
        return self._sess.run(*a, **kwa)

    def partial_run_setup(self, *a, **kwa):
        raise RuntimeError("Not supported in session wrapper")

    def partial_run(self, *a, **kwa):
        raise RuntimeError("Not supported in session wrapper")

    def make_callable(self, *a, **kwa):
        raise RuntimeError("Not supported in session wrapper")

    def as_default(self):
        return ops.default_session(self)

    def __getattr__(self, attr):
        return getattr(self._sess, attr)

    def __enter__(self):
        if self._default_session_context_manager is None:
            self._default_session_context_manager = self.as_default()
        return self._default_session_context_manager.__enter__()

    def __exit__(self, *exc):
        self._default_session_context_manager.__exit__(*exc)

    def __del__(self):
        self._sess.__del__()


class ProfilableSessionWrapper(SessionWrapper):
    def __init__(self, session, log_dir, skip_first_nruns=0, profile_level=0):
        super(ProfilableSessionWrapper, self).__init__(session)

        self.run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        self.run_metadata = tf.RunMetadata()
        self.run_counter = 0
        self.nruns_threshold = skip_first_nruns
        self.profile_level = profile_level

        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self.op_log = None
        tf.profiler.write_op_log(
            tf.get_default_graph(),
            log_dir=log_dir,
            op_log=self.op_log,
            run_meta=self.run_metadata
            )

    def _write_log(self):
        print("* --------------------------------------", file=sys.stderr)
        print("* RUN: %d" % self.run_counter, file=sys.stderr)

        # 1. Fetch memory usage and timing stat
        time_stat_options = model_analyzer.PRINT_ALL_TIMING_MEMORY
        time_stat_options['output'] = 'file:outfile=%s/time_stat.run_%d.txt' % (self.log_dir, self.run_counter)
        time_stat_options['select'] = ['device', 'micros', 'bytes']
        time_stat_options['order_by'] = 'micros'
        tf.profiler.profile(
            tf.get_default_graph(),
            run_meta=self.run_metadata,
            op_log=self.op_log,
            options=time_stat_options
            )

        # 2. Create timeline.json file. It can be load in chrome://tracing
        time_data = timeline.Timeline(self.run_metadata.step_stats)
        trace = time_data.generate_chrome_trace_format(show_memory=True)
        timeline_fname = '%s/timeline.run_%d.json' % (self.log_dir, self.run_counter)
        with open(timeline_fname, 'w') as f:
            f.write(trace)

        # 3. Get peak memory
        mem_timelines = self._build_memory_timelines()
        peak_memory = self._compute_peak_memory(mem_timelines)
        print("Peak memory: %s" % str(peak_memory), file=sys.stderr)

        # 4. Print memory timelines
        for allocator, tl in mem_timelines.items():
            memory_fname = '%s/memory.%s.run_%d.txt' % (self.log_dir, allocator, self.run_counter)
            with open(memory_fname, 'w') as f:
                print("ts,node_name,bytes_in_use,live_bytes", file=f)
                for r in tl:
                    print("%d,%s,%d,%d" % (r.ts, r.node_name, r.bytes_in_use, r.live_bytes), file=f)

    def run(self, fetches, feed_dict=None, options=None, run_metadata=None):
        do_profile = self.run_counter >= self.nruns_threshold and self.profile_level >= get_profile_level()
        result = super(ProfilableSessionWrapper, self).run(
            fetches, feed_dict,
            options=self.run_options if do_profile else None,
            run_metadata=self.run_metadata if do_profile else None
            )
        # For earch invocation of `run()` or `eval()` methods dump log to new file
        if do_profile and lib.ops.mpi.is_master():
            self._write_log()
        self.run_counter += 1
        return result

    def _compute_peak_memory(self, mem_timelines):
        res = {}
        for k, tl in mem_timelines.items():
            res[k] = max([r.bytes_in_use for r in tl])
        return res

    def _build_memory_timelines(self):
        timelines = {}

        for dev in self.run_metadata.step_stats.dev_stats:
            for node in dev.node_stats:
                ts = node.all_start_micros
                for mem in node.memory:
                    if mem.allocator_name not in timelines:
                        timelines[mem.allocator_name] = []
                    timelines[mem.allocator_name].append(MemTimelineRecord(ts, node.node_name, mem.allocator_bytes_in_use, mem.live_bytes))

        for tl in timelines.values():
            tl.sort()

        return timelines

    def _simplify_device_name(self, device_name):
        return '/' + device_name.split('device:')[1]
