## Analyzing Source and Target Contribution to NMT Predictions


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
   * [Evaluate LRP](#evaluate-lrp)
   * [Models](#models)


# What is this about?

	
In NMT, each prediction is based on two types of context: the source and the prefix of the target sentence. We show how to evaluate the relative contributions of source and target to NMT predictions and find that:

* models suffering from exposure bias are more prone to over-relying on target history (and hence to hallucinating) than the ones where the exposure bias is mitigated;

* models trained with more data rely on the source more and do it more confidently;

* the training process is non-monotonic with several distinct stages.


# Evaluating LRP



# Models
