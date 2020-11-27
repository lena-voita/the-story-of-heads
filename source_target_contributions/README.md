# Analyzing Source and Target Contribution to NMT Predictions


<img src="../resources/src_dst_main.gif" 
	title="paper logo" width="400" align="right"/>

In this part, we discuss experiments from the paper [Analyzing the Source and Target Contributions to Predictions in Neural Machine Translation](https://arxiv.org/pdf/2010.10907.pdf).

For more details, look at the [blog post](https://lena-voita.github.io/posts/source_target_contributions_to_nmt.html)!
		
#### Bibtex
```
@misc{voita2020analyzing,
      title={Analyzing the Source and Target Contributions to Predictions in Neural Machine Translation}, 
      author={Elena Voita and Rico Sennrich and Ivan Titov},
      year={2020},
      booktitle = "{{arXiv}:2010.10907}",
      url = "https://arxiv.org/abs/2010.10907",
}
```

Table of Contents
=================

   * [What is this about?](#what-is-this-about)
   * [Models](#models)
      * [Word Dropout](#word-dropout)
      * [Minimum Risk Training](#minimum-risk-training)
   * [Evaluating LRP](#evaluating-lrp)


# What is this about?

<img src="../resources/intro_large-min.png" 
	title="intro" />
	
In NMT, each prediction is based on two types of context: the source and the prefix of the target sentence. We show how to evaluate the relative contributions of source and target to NMT predictions and find that:

* models suffering from exposure bias are more prone to over-relying on target history (and hence to hallucinating) than the ones where the exposure bias is mitigated;

* models trained with more data rely on the source more and do it more confidently;

* the training process is non-monotonic with several distinct stages.


# Models

For the general training pipeline, see the [explanation in the main README](../README.md). For the experiments with LRP, you have to set the model to `transformer_lrp` as follows:

```
params=(
...
--model lib.task.seq2seq.models.transformer_lrp.Model
...)
```

## Baseline

Use the [train_baselin.sh](../scripts/train_baseline.sh) script and set the model to the one I mentioned above.

## Word Dropout

To use word dropout on the source side, add the following options to the default problem:
```
params=(
    ...
    --problem lib.task.seq2seq.problems.default.DefaultProblem
    --problem-opts '{'"'"'inp_word_dropout'"'"': 0.1, '"'"'word_dropout_method'"'"': '"'"'random_word'"'"',}'
    ...)
```
For the dropout on the target side, replace `inp_word_dropout` with `out_word_dropout`.

## Minimum Risk Training


# Evaluating LRP


Here are the useful notebooks:
* [1_Load_model_and_evaluate_LRP](./1_Load_model_and_evaluate_LRP.ipynb) - load a model and evaluate LRP for a dataset;

* [2_Load_LRP_results_and_build_graphs](./2_Load_LRP_results_and_build_graphs.ipynb) - load LRP results we've built before and plot the graphs.
