Ensemble Model Optimization (CLEAVER)
==========

CLEAVER is the first optimization method which has been implemented inside QuickRank. It is a post-learning optimizer which acts on a model composed by an ensemble of weak rankers, by pruning the ensemble and re-weighting the remaining weak rankers, improving consequently its efficiency, without hindering its effectiveness. Since the model has to be composed by an ensemble of weak rankers, traditional algorithms which use forest of decision tree are ok (MART, LAMBDAMART, RANDOM FOREST, etc). 

CLEAVER implements various pruning strategies. Let be *n* the number of weak rankers and *k* the number of weak rankers to prune out. The pruning strategies are the following:

- **RANDOM**: The *k* weak rankers are pruned totally at random (each one is considered equally important).
- **LAST**: The last *k* weak rankers are pruned. The conjecture is that initial weak rankers are more precise due to boosting algorithms.
- **SKIP**: Two close weak rankers are considered more similar in terms of discriminative power. The strategy keep one every *[n/(n-k)]* weak rankers.
- **LOW_WEIGHTS**: The weights associated with each tree are optimized using line search (see below). The *k* weak rankers with lowest weight are then removed.
- **SCORE_LOSS**: The normalized contribution of each weak rankers to the final score is computed with *s_i(q,d) / S(q,d)* and the *k* weak rankers that contribute less are removed.
- **QUALITY_LOSS**: For each weak rankers, the average quality drop (in terms of loss function) is computed by scoring the documents without that tree. Then the *k* weak rankers with the smallest quality drop are removed.
- **QUALITY_LOSS_ADV**: At each iteration, the average quality drop (in terms of loss function) is computed by scoring the documents without each of the weak rankers. Then, the weak ranker with the smallest quality drop is removed. The process is repeated until *k* weak rankers are pruned out.  

In order to optimize the weights associated with each weak rankers, CLEAVER 
makes use of a greedy Line Search process which iteratively tries to find the 
best weight vector. In addition to the traditional Line Search behaviour, in 
QuickRank 

Since QuickRank is a framework with efficiency in mind, CLEAVER makes use of some "optimizations" in order to be as fast as possible in performing the optimization on the model. To do that, it transforms the original dataset in a new dataset with the same rows of the original ones (query-document pairs), but with columns representing the partial score of each weak rankers on these pairs (i.e., the new dataset is model dependent, which means it has to be recomputed for different models and for different datasets).

Being CLEAVER an optimization method inside QuickRank, it can be ran in pipeline with the training process or as a standalone process. The former option can be executed by specifying the options for training the model plus the ones for optimizing it. The framework will take care of training the model, transforming the dataset as described above, executing the optimization process and writing the final models on files (the original trained model plus the optimized model) as well as the saving the transformed dataset (if the relative options are passed via command line parameters).  

```
./bin/quicklearn \
  --algo LAMBDAMART \
  --train quickranktestdata/msn1/msn1.fold1.train.5k.txt \
  --valid quickranktestdata/msn1/msn1.fold1.vali.5k.txt \
  --model-out lambdamart-model.xml \
  --opt-algo CLEAVER \
  --opt-method QUALITY_LOSS \
  --opt-model optmization-model.xml \
  --opt-algo-model lambdamart-optimized-model.xml \
  --pruning-rate 0.5 \
  --with-line-search \
  --num-samples 10 \
  --window-size 1 \
  --reduction-factor 0.95 \
  --max-iterations 100 \
  --max-failed-valid 20 \
  --train-partial partial-score-train.txt \
  --valid-partial partial-score-vali.txt
```

The above command will train a LAMBDAMART model with default parameters (but you can personalize the model as usual by passing additional parameters) and save it on file `lambdamart-model.xml`. Then the dataset transformation will take place, and the resulting training and validation datasets are saved respectively on files `partial-score-train.txt` and `partial-score-vali.txt`. The optimization process starts, and at the end both the optimization model (which is a special model that can be used for obtaining the optimized model or re-run the optimization process) as well as the optimized model (which is the original model but without some of the weak rankers, that has been pruned out from the ensemble, and with optimized weights) are saved respectively on files `optmization-model.xml` and `lambdamart-optimized-model.xml`.

Running CLEAVER as a standalone process requires to separately train a model. This is done the standard way (described [here](/README.md)). Then optimizing it is done very similarly to how is described above, except for the part where the LtR model is loaded from the file (in this example the model is loaded from `lambdamart-model.xml`):

```
./bin/quicklearn \
  --model-in lambdamart-model.xml \
  --train quickranktestdata/msn1/msn1.fold1.train.5k.txt \
  --valid quickranktestdata/msn1/msn1.fold1.vali.5k.txt \
  --opt-algo CLEAVER \
  --opt-method QUALITY_LOSS \
  --opt-model optmization-model.xml \
  --opt-algo-model lambdamart-optimized-model.xml \
  --pruning-rate 0.5 \
  --with-line-search \
  --num-samples 10 \
  --window-size 1 \
  --reduction-factor 0.95 \
  --max-iterations 100 \
  --max-failed-valid 20 \
  --train-partial partial-score-train.txt \
  --valid-partial partial-score-vali.txt
```

Since it is popular to try different pruning level and different pruning strategies, the best (most efficient) way to do so is by transforming the training and validation datasets into their partial score format, and in running the line search process which is used by some pruning strategies to have the initial weights assignment. This way CLEAVER will avoid to recompute these tasks for every experiment. To execute such a tasks the first thing to do is to transform the datasets into the partial score format (to do for both the training and validation datasets).

```
./bin/quicklearn \
  --model-in lambdamart-model.xml \
  --test quickranktestdata/msn1/msn1.fold1.train.5k.txt \
  --scores partial-score-train.txt \
  --detailed
```

```
./bin/quicklearn \
  --model-in lambdamart-model.xml \
  --test quickranktestdata/msn1/msn1.fold1.valid.5k.txt \
  --scores partial-score-vali.txt \
  --detailed
```

Then to execute the LINESEARCH process above these datasets:

```
./bin/quicklearn \
  --algo LINESEARCH \
  --train partial-score-train.txt \
  --valid partial-score-vali.txt \
  --num-samples 10 \
  --window-size 10   \
  --reduction-factor 0.95 \
  --max-iterations 100 \
  --max-failed-valid 20
  --model-out line-search-model.xml
```

Having done these tasks, it is now possible to efficiently test various pruning strategies and pruning levels:

```
./bin/quicklearn \
  --model-in lambdamart-model.xml \
  --train-partial partial-score-train.txt \
  --valid-partial partial-score-vali.txt \
  --opt-algo CLEAVER \
  --opt-method QUALITY_LOSS \
  --opt-model optmization-model.xml \
  --opt-algo-model lambdamart-optimized-model.xml \
  --pruning-rate 0.5 \
  --with-line-search \
  --line-search-model line-search-model.xml
```

Experiments Reproducibility
-------
In order to reproduce the experiments reported in the paper where CLEAVER has
been presented, in this section we report the values of the parameters adopted:

- Optimization method: QUALITY_LOSS
- Num samples: 20
- Window size: 2
- Reduction factor: 0.95
- Max iterations: 100
- Max failed valid: 20
- Adaptive reduction factor: true 
- Pruning rate: 10 - 90% 

The adaptive parameter in CLEAVER adapts the window size depending from the gain
observed in the previous iteration of the line search process. It has been 
used to fasten the greedy search process and it works by multiplying the 
previous window size by a factor in the range [0.5 - 2.0].
 
At the following link it is also possible to download the models as well as the 
training and test output: [CLEAVER.tar.gz](http://hpc.isti.cnr.it/~trani/CLEAVER.tar.gz)  

Acknowledgements
-------

**CLEAVER** has beed presented in the following paper:
 - C. Lucchese, F. M. Nardini, S. Orlando, R. Perego, F. Silvestri, S. Trani.
 **Post-Learning Optimization of Tree Ensembles for Efficient Ranking**.
 *ACM SIGIR Conference on Research and Development in Information Retrieval*, (2016).
 [LINK](http://dx.doi.org/10.1145/2911451.2914763).

License
-------
© Contributors, 2016. Licensed under an [Reciprocal Public License (RPL-1.5)](https://opensource.org/licenses/RPL-1.5).
