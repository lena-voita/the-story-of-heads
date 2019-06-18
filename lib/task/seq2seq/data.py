import sys
import random
from sortedcontainers import SortedList
import numpy as np
import math
import itertools
import tensorflow as tf

from lib.data import pad_seq_list


def srclen(item):
    return item[0].count(' ') + 1


def dstlen(item):
    return item[1].count(' ') + 1


def maxlen(item):
    return max(srclen(item), dstlen(item))


def sumlen(item):
    return srclen(item) + dstlen(item)


def form_batches(data, batch_size):
    seq = iter(data)
    done = False
    while not done:
        batch = []
        for _ in range(batch_size):
            try:
                batch.append(next(seq))
            except StopIteration:
                done = True
        if batch:
            yield batch


def locally_sorted_by_len(seq, window, weight_func=maxlen, alterate=False):
    reverse = False
    for batch in form_batches(seq, window):
        batch = sorted(batch, key=weight_func, reverse=reverse)
        for x in batch:
            yield x
        if alterate:
            reverse = not reverse


def form_adaptive_batches(data, batch_len, batch_size_max=0):
    seq = iter(data)
    prev = []
    max_len = 0
    done = False
    while not done:
        batch = prev
        try:
            while True:
                item = next(seq)
                max_len = max(max_len, maxlen(item))
                if (len(batch) + 1) * max_len > batch_len or (batch_size_max and len(batch) >= batch_size_max):
                    prev, max_len = [item], maxlen(item)
                    break
                batch.append(item)
        except StopIteration:
            done = True
        if batch:
            yield batch


def form_adaptive_batches_windowed(data, weight_func=maxlen, max_size=5000, split_len=10000, batch_size_max=0):
    rng = random.Random(42)
    buf = []
    last_chunk = []
    reverse = False
    for p in data:
        if len(buf) >= split_len:
            # Last chunk may contain fewer sentences than others - let's return in to the miller
            buf += last_chunk

            buf = sorted(buf, key=weight_func, reverse=reverse)
            chunks = list(form_adaptive_batches(buf, max_size, batch_size_max=batch_size_max))

            last_chunk = chunks.pop()
            buf = []

            reverse = not reverse

            rng.shuffle(chunks)
            for chunk in chunks:
                yield chunk
        buf.append(p)

    buf += last_chunk
    buf = sorted(buf, key=weight_func, reverse=reverse)
    chunks = list(form_adaptive_batches(buf, max_size, batch_size_max=batch_size_max))
    rng.shuffle(chunks)
    for chunk in chunks:
        yield chunk


def batch_cost(x):
    return len(x) + 2 * (max(len(i[0]) for i in x) + max(len(i[1]) for i in x))


def load_parallel(src, dst, cycle=False):
    # Load data.
    for i in itertools.count():
        if i > 0 and not cycle:
            break
        data = zip(open(src), open(dst))
        for l, r in data:
            yield l.rstrip('\n'), r.rstrip('\n')


def filter_by_len(data, max_srclen=sys.maxsize, max_dstlen=sys.maxsize, batch_len=None):
    def item_ok(item):
        ok = (srclen(item) <= max_srclen and dstlen(item) <= max_dstlen)
        if batch_len is not None:
            return ok and maxlen(item) <= batch_len
        return ok

    return filter(item_ok, data)


# ======= Random block reader =====================
def _read_file_part(fd, file_size, part_id, nparts):
    begin_pos = (part_id * file_size) // nparts
    end_pos = ((part_id + 1) * file_size) // nparts
    fd.seek(begin_pos)

    if part_id > 0:
        _prefix = fd.readline()  # skip first line

    current_pos = fd.tell()  # get offset
    if current_pos < end_pos:
        body = fd.readlines(end_pos - current_pos)
    elif current_pos == end_pos and end_pos < file_size:
        body = [fd.readline()]
    else:
        return []

    return body


def _grouper(data, n=5):
    pool = []
    for item in data:
        if len(pool) == n:
            yield pool
            pool = []
        pool.append(item)
    yield pool


def random_block_reader(fname, part_size=64 * 1024, parallel=5, infinite=False, seed=42, encoding='utf-8'):
    fd = open(fname, 'rb')
    fd.seek(0, 2)
    file_size = fd.tell()

    nparts = math.ceil(file_size / part_size)
    rng = np.random.RandomState(seed)
    if infinite:
        part_ids = (rng.randint(0, nparts) for _ in itertools.count())
    else:
        part_ids = np.arange(nparts)
        rng.shuffle(part_ids)

    for group in _grouper(part_ids, parallel):
        lines = []
        for part_id in group:
            part_lines = _read_file_part(fd, file_size, part_id, nparts)
            lines += part_lines

        rng.shuffle(lines)

        for line in lines:
            yield line.decode(encoding).rstrip('\n')

    fd.close()


# ======= Adaptive batches with sorted data =======
class FastSplitter2d:
    def __init__(self, max_size=5000, chunk_count=5):
        self.max_size = max_size
        self.max_x = 0
        self.points = SortedList(key=lambda p: -p[1])
        self.chunk_count = chunk_count

    def add_to_pack(self, p):
        self.max_x = max(self.max_x, p[0])
        new_pos = self.points.bisect_right(p)
        self.points.insert(new_pos, p)

        offset = 0
        bs_vec = []
        while offset < len(self.points):
            bs = self.max_size // (self.max_x + self.points[offset][1])
            bs = min(len(self.points) - offset, bs)
            bs_vec.append(bs)
            offset += bs

        return new_pos, bs_vec

    def make_chunk_gen(self, points):
        prev_bs_vec = [0]
        for p in sorted(list(points), key=lambda p: p[0], reverse=True):
            new_pos, bs_vec = self.add_to_pack(p)

            if len(bs_vec) > len(prev_bs_vec):
                if len(prev_bs_vec) >= self.chunk_count:
                    self.points.pop(new_pos)
                    offset = 0
                    for sz in prev_bs_vec:
                        yield self.points[offset:offset + sz]
                        offset += sz
                    self.points.clear()
                    self.points.add(p)
                    prev_bs_vec = [1]
                    self.max_x = p[0]
            prev_bs_vec = bs_vec
        offset = 0
        for sz in prev_bs_vec:
            yield self.points[offset:offset + sz]
            offset += sz


def _in_conv(item, lead_inp_len=False):
    x_len = item[0].count(' ') + 1
    y_len = item[1].count(' ') + 1
    if lead_inp_len:
        return item.__class__((x_len, y_len)) + item
    else:
        return item.__class__((y_len, x_len)) + item


def _out_chunk_conv(chunk):
    return [item[2:] for item in chunk]


def form_adaptive_batches_split2d(data, max_size=5000, split_len=10000, chunk_count=5, lead_inp_len=False):
    rng = random.Random(42)
    buf = []
    for p in data:
        if len(buf) >= split_len:
            splitter = FastSplitter2d(max_size=max_size, chunk_count=chunk_count)
            chunks = list(splitter.make_chunk_gen(buf))
            rng.shuffle(chunks)
            for chunk in chunks:
                if len(chunk) == 0:
                    print("SPLIT2D: empty chunk", file=sys.stderr)
                    continue
                yield _out_chunk_conv(chunk)
            buf = []

        buf.append(_in_conv(p, lead_inp_len=lead_inp_len))

    splitter = FastSplitter2d(max_size=max_size, chunk_count=chunk_count)
    chunks = list(splitter.make_chunk_gen(buf))
    rng.shuffle(chunks)
    for chunk in chunks:
        if len(chunk) == 0:
            print("SPLIT2D: empty chunk", file=sys.stderr)
            continue
        yield _out_chunk_conv(chunk)


## ============================================================================
#                               Integration

def words_from_line(line, voc, bos=0, eos=1):
    line = line.rstrip('\n')
    words = [token for token in line.split(' ') if token]
    return voc.words([voc.bos]) * bos + words + voc.words([voc.eos]) * eos


def words_from_ids(ids, voc):
    return [
        word if (id not in [voc.bos, voc.eos]) else None
        for id, word in zip(ids, voc.words(ids))
    ]


def lines2ids(lines, voc, **kwargs):
    # Read as-is, without padding.
    ids_all = []
    for line in lines:
        words = words_from_line(line, voc, **kwargs)
        ids = voc.ids(words)
        ids_all.append(ids)

    # Pad and transpose.
    ids_all, ids_len = pad_seq_list(ids_all, voc.eos)
    return ids_all, ids_len


def make_batch_data(batch, inp_voc, out_voc, force_bos=False, **kwargs):
    inp_lines, out_lines = zip(*batch)
    inp, inp_len = lines2ids(inp_lines, inp_voc, bos=int(force_bos))
    out, out_len = lines2ids(out_lines, out_voc, bos=int(force_bos))

    batch_data = dict(
        inp=np.array(inp, dtype=np.int32),
        inp_len=np.array(inp_len, dtype=np.int32),
        out=np.array(out, dtype=np.int32),
        out_len=np.array(out_len, dtype=np.int32))

    return batch_data


def make_batch_placeholder(batch_data):
    batch_placeholder = {
        k: tf.placeholder(v.dtype, [None] * len(v.shape))
        for k, v in batch_data.items()}
    return batch_placeholder


class BatchIndexer:
    pass


