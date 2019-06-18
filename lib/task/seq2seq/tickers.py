import os
import sys
import tensorflow as tf

from ...train.tickers import DistributedTicker, _IsItTimeYet
import lib
from .bleu import Bleu

# - TranslateTicker - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def unbpe(sent):
    return sent.replace(' `', '')


class TranslateTicker(DistributedTicker):
    """
    - Translate devset once in a while.
    - Print BLEU to stderr after each translation.
    """
    def __init__(self, model_name, devset, name='Dev', every_steps=None, every_minutes=None, initial=False, folder=None,
                 suffix=None, device=None, parallel=True):
        self.model_name = model_name
        self.devset = devset
        self.every_steps = every_steps
        self.every_minutes = every_minutes
        self.folder = folder
        self.name = name
        self.initial = initial
        self.device = device
        self.parallel = parallel
        self.suffix = suffix if suffix is not None else model_name
        if self.suffix:  # add underscore if we add suffix
            self.suffix = '_' + self.suffix

    def on_started(self, context):
        self.devset_batches = list(self.devset)
        self.context = context
        self.model = context.get_model(self.model_name)

        self.bleu = tf.placeholder(tf.float32)
        self.translations = tf.placeholder(tf.string, shape=[None])

        self.summary = tf.summary.merge([
            tf.summary.scalar(("%s/BLEU" % self.name) + self.suffix, self.bleu),
            tf.summary.text(("%s/Translations" % self.name) + self.suffix, self.translations)])

        self.is_it_time_yet = _IsItTimeYet(
            context, self.every_steps, self.every_minutes)

        # Score devset after initialization if option passed (and we are not loading some non-init checkpoint)
        if self.initial and context.get_global_step() == 0:
            self._score()

    def after_train_batch(self, ingraph_result):
        if self.is_it_time_yet():
            self._score()

    def _score(self):
        if lib.ops.mpi.is_master():
            print('Translating', end='', file=sys.stderr, flush=True)

        translations = None

        if self.parallel or lib.ops.mpi.is_master():
            translations = []
            with tf.device(self.device) if self.device is not None else lib.util.nop_ctx():
                for batch in self.devset_batches:
                    trans = self.model.translate_lines([line[0] for line in batch])
                    for index in range(len(batch)):
                        src = unbpe(batch[index][0])
                        ethalon = unbpe(batch[index][1])
                        translations.append(src + '\t' + ethalon + '\t' + unbpe(trans[index]))

            if self.parallel:
                translations = lib.ops.mpi.gather_obj(translations)
                if translations is not None:
                    translations = [x for t in translations for x in t]

        if translations is not None:
            # compute BLEU only on the master

            if self.folder is not None:
                global_step = self.context.get_global_step()
                self._dump_translations(
                    translations,
                    fname='translations{}_{}.txt'.format(self.suffix, global_step)
                )

            bleu = Bleu()
            for translation in translations:
                src, ethalon, trans = translation.split('\t')
                bleu.process_next(trans, [ethalon])
            bleu_value = 100 * (bleu.total()[0])

            print('BLEU %f' % bleu_value, file=sys.stderr, flush=True)

            summary = tf.get_default_session().run(self.summary, feed_dict={self.bleu: bleu_value,
                                                                            self.translations: translations})

            self.context.get_summary_writer().add_summary(summary, self.context.get_global_step())

    def _dump_translations(self, translations, fname):
        if not os.path.isdir(self.folder):
            os.mkdir(self.folder)
        fout = open(os.path.join(self.folder, fname), 'w')
        for translation in translations:
            print(translation, file=fout)
