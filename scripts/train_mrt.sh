#!/bin/bash

REPO_DIR="../" # insert the dir to the the-story-of-heads repo
DATA_DIR="../" # insert your datadir

NMT="${REPO_DIR}/scripts/nmt.py"

# path to preprocessed data (tokenized, bpe-ized)
train_src="${DATA_DIR}/train.src"
train_dst="${DATA_DIR}/train.dst"
dev_src="${DATA_DIR}/dev.src"
dev_dst="${DATA_DIR}/dev.dst"

# path where results will be stored
model_path="./build"
mkdir -p $model_path

# make vocabularies
if [ ! -f $model_path/src.voc ]; then
  echo "Creating source language vocabulary"
  $NMT mkvoc --text $train_src --outvoc $model_path/src.voc --n-words=1000000
  # n-words is the maximum number of tokens in vocabulary
  # in practice it is unlikely to be reached if you are using BPE subwords
fi

if [ ! -f $model_path/dst.voc ]; then
  echo "creating destination language vocabulary"
  $NMT mkvoc --text $train_dst --outvoc $model_path/dst.voc --n-words=1000000
fi


# shuffle data
shuffle_seed=42

get_random_source()
{
  openssl enc -aes-256-ctr -pass pass:"$1" -nosalt </dev/zero 2>/dev/null
}


if [ ! -f $model_path/train.src.shuf ]; then
  echo "Shuffling train src"
  shuf -o $model_path/train.src.shuf --random-source=<(get_random_source $shuffle_seed) $train_src
fi
if [ ! -f $model_path/train.dst.shuf ]; then
  echo "Shuffling train dst"
  shuf -o $model_path/train.dst.shuf --random-source=<(get_random_source $shuffle_seed) $train_dst
fi


# maybe add openmpi wrapper
RUN_NMT=$(/usr/bin/env python3 -c "
import os, sys, tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
num_gpus = sum(x.device_type == 'GPU' for x in tf.Session().list_devices())
if num_gpus > 1:
    sys.stdout.write('mpirun --allow-run-as-root --host {} {}'.format(','.join(['localhost'] * num_gpus), '$NMT'))
else:
    sys.stdout.write('$NMT')
")

# Model hp (the same as in the original Transformer-base)
# hp are split into groups:
#      main model hp,
#      minor model hp (probably you do not want to change them)
#      regularization and label smoothing
#      inference params (beam search with a beam of 4)

MODEL_HP=$(/usr/bin/env python3 -c '

hp = {
     "num_layers": 6,
     "num_heads": 8,
     "ff_size": 2048,
     "ffn_type": "conv_relu",
     "hid_size": 512,
     "emb_size": 512,
     "res_steps": "nlda",

     "rescale_emb": True,
     "inp_emb_bias": True,
     "normalize_out": True,
     "share_emb": False,
     "replace": 0,

     "relu_dropout": 0.1,
     "res_dropout": 0.1,
     "attn_dropout": 0.1,
     "label_smoothing": 0.1,

     "translator": "ingraph",
     "beam_size": 4,
     "beam_spread": 3,
     "len_alpha": 0.6,
     "attn_beta": 0,
    }

print(end=repr(hp))
')

params=(
    --folder $model_path
    --seed 42

    --train-src $model_path/train.src.shuf
    --train-dst $model_path/train.dst.shuf
    --dev-src $dev_src
    --dev-dst $dev_dst

    --ivoc $model_path/src.voc
    --ovoc $model_path/dst.voc

    # Model you want to train
    --model lib.task.seq2seq.models.transformer_lrp.Model
    # Model hp (specified above)
    --hp "`echo $MODEL_HP`"

    # Problem, i.e. how to train your model (loss function).
    --problem lib.task.seq2seq.problems.mrt.MRTProblem
    # Problem options.
    # For the MRTProblem, we need to specify 'num_hypos'
    --problem-opts '{'"'"'num_hypos'"'"': 50,}'


    # Starting checkpoint.
    # If you prune head starting from a trained model (as we did), you have to specify a starting checkpoint.
    --pre-init-model-checkpoint 'dir_to_your_trained_baseline_checkpoint.npz'
    #                             ^---YOU HAVE TO CHANGE THIS

    # Maximum number of tokens in a sentence.
    # Sentences longer than this will not participate in training.
    --max-srclen 200
    --max-dstlen 200

    # How to form batches.
    # The only thing you have to be careful about is batch-len:
    # is has to be about 16000 in total. Here is 4000 for 4 gpus: 4 * 4000 in total.
    --batch-len 70
    #            ^---YOU MAY WANT TO CHANGE THIS
    --batch-maker adaptive_windowed
    --shuffle-len 100000
    --batch-shuffle-len 10000
    --split-len 200000
    --maxlen-quant 1
    --maxlen-min 8

    # Optimization.
    # This is the optimizer used in the original Transformer.
    --optimizer lazy_adam

    # Optimizer opts: virtual batch.
    # sync_every_steps=4 means that you accumulate gradients for 4 steps before making an update.
    # This is equivalent to having 'sync_every_steps' gpus.
    # The actual batch size will be then batch-len * sync_every_steps
    # (or batch-len * num_gpus if you are using the first version of optimizer-opts)
    --optimizer-opts '{'"'"'beta1'"'"': 0.9, '"'"'beta2'"'"': 0.998,
                       '"'"'sync_every_steps'"'"': 400, '"'"'average_grads'"'"': True, }'
    #                                               ^---NOTE THAT THIS IS LARGE

    # Learning rate schedule.
    # This is the usual Transformer learning rate schedule.
    --learning-rate 4.0
    --learning-rate-stop-value 1e-08
    --decay-steps 16000
    --decay-policy t2t_noam

    # After 900 batches, we changed the schedule to constant. See below.
    # --learning-rate 5e-5
    # --decay-policy constant

    # How long to train.
    --num-batches 900
    # After 900 batches, change the learning rate schedule and train for 500 more batches.

    # Checkpoints.
    # How often to make a checkpoint
    --checkpoint-every-steps 50
    # How many checkpoints you want to keep.
    --keep-checkpoints-max 10
    #                       ^---YOU MAY WANT TO CHANGE THIS

    # How often to score dev set (and put a dot on your tensorboard)
    --score-dev-every 25

    # BLEU on your tensorboard.
    # This says that you want to see BLEU score on your tensorboard.
    --translate-dev
    # How often to translate dev and add this info to your tensorboard.
    --translate-dev-every 50

    # This argument has to passed last.
    # It controls that nmt.py has received all your arguments
    --end-of-params
)

$RUN_NMT train "${params[@]}"


