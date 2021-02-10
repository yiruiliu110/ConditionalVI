# 
# CATVI: Conditional and Adaptively Truncated VariationalInference for Hierarchical Bayesian Nonparametric Models

This repository contains the implementation for the paper "CATVI: Conditional and Adaptively Truncated VariationalInference for Hierarchical Bayesian Nonparametric Models" (ICML 2021) in Python.

## Summary of the paper

Current variational inference methods for hierar-chical Bayesian nonparametric models can nei-ther characterize the correlation structure amonglatent variables due to the mean-field setting, norinfer the true posterior dimension because of theuniversal  truncation.   To  overcome  these  limi-tations,  we  propose  the  conditional  and  adap-tively   truncated   variational   inference   method(CATVI) by maximizing the nonparametric ev-idence lower bound and integrating Monte Carlosampling into the stochastic variational inferenceframework.   CATVI  enjoys  several  advantagesover  traditional  variational  inference  methods,including  a  smaller  divergence  between  varia-tional  and  true  posteriors,  reduced  risk  of  un-derfitting or overfitting, and improved predictionaccuracy.    The  empirical  study  on  three  largedatasets,arXiv,New York TimesandWikipedia,reveals that CATVI applied in Bayesian nonpara-metric  topic  models  substantially  outperformsthe competitors in terms of lower perplexity andmuch clearer topic-words clustering.
<img align="center" src="diag.png" alt="drawing" width="600">



## Requirements

Change your working directory to this main folder.





## Contributing

All contributions welcome! All content in this repository is licensed under the MIT license.
