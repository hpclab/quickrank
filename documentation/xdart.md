Dart and X-Dart: Blending Dropout and Pruning 
==========

Traditional learning to rank algorithms, like MART and LAMBDAMART, suffers from 
over-specialization. They build the model iteratively by adding a tree at a 
time trying to minimize the cost function adopted. The trees added at later 
iterations tend however to impact the prediction of only a few instances and 
to give a negligible contribution to the  nal score of the remaining 
instances. This has two important negative effects: i) it negatively a ects 
the performance of the model on unseen data, and ii) it makes the learned 
model over-sensitive to the contributions of a few initial trees.

To overcome these two limitations, in 2015 Rashmi and Gilad-Bachrach proposed
DART [1], a new algorithm that borrows the concept of dropout from neural 
networks. Dropout has been adapted to ensemble of trees by muting complete 
trees as opposed to mute neurons. At each iteration, a dropout set D of trees 
is randomly selected (size of D is a fraction of the current ensemble size) and 
their contributions for the ramaining of the current iteration are ignored. 
Then a new tree T is learnt by using a given base tree-learning algorithm A 
(e.g., using MART or LAMBDAMART). This perturbation strategy (training a new 
tree by muting several trees in the ensemble) allows to reduce the risk of 
over-fitting. The algorithm finally adds back the dropout set D together with 
the new tree T to the pruned model after a normalization step. DART has been 
proved to remarkably outperform MART and LAMBDAMART in terms of effectiveness.

On the other hand, driven by efficiency reasons recently Lucchese et al. 
proposed [CLEaVER](cleaver.md) [2], a post-learning optimization framework 
for MART models. The goal is in this case improving the efficiency of the 
learned model at document scoring time without affecting ranking quality.

Inspired by these two orthogonal works, in 2017 Lucchese et al. proposed an 
extension of Dart, namely X-DART, that improves over DART by borrowing 
from CLEaVER the tree pruning strategy, eventually providing more robust and 
compact ranking models.

DART and X-DART have been implemented inside QuickRank. Several parameters drive
the behaviour of the training process, affecting the effectiveness and the 
efficiency of the resulting model. Below a description of the various parameters
and the command line to use for training/test DART and X-DART models.

X-DART implements various sampling strategies (**sample-type** parameter name) 
for selecting the set D of trees to mute. Let be *n* the number of trees in 
the ensemble and *k* the number of weak rankers to select (mute). The 
sampling strategies are the following:

- **UNIFORM**: The *k* trees are selected totally at random (each one is
 considered equally important).
- **WEIGHTED**: Generate a random permutation with different probability
 for each element to be selected (according to the trees' weight). Then 
 according to these probabilities selects *k* trees.
- **WEIGHTED_INV**: As above but reverting the probabilities (i.e., the 
probability of each tree is 1-prob as computed above)  
- **TOP_FIFTY**: Behave like *UNIFORM* but selects the trees to mute only from 
the first half of the ensemble.
- **CONTR**: Behave like *WEIGHTED* but using the average contribution of 
each tree (w/o considering the weight of each tree) to the total score for 
computing the probaiblity.  
- **CONTR_INV**: Like *CONTR* but reverting the probabilities 
- **WCONTR**: Like *CONTR* but computing the probabilities taking into 
account also the weight of each tree (i.e., the contribution of each tree is 
the weight of the tree times the average contribution). 
- **WCONTR_INV**: Like *WCONTR* but reverting the probabilities.
- **TOP_WCONTR**: The *k* trees with the highest average weighted 
contribution are selected.  
- **LESS_WCONTR**: As *TOP_WCONTR* but selecting the trees with the lowest 
average weighted contribution. 

X-DART implements also various normalization strategies (**normalize-type** 
parameter name) for modifying the weights of both the last learned tree and the 
trees in the dropout set D when they are put back in the ensemble. Let 
*shrinkage* be the learning rate of the base algorithm, *k* the number of 
dropped trees, *w_t_prune* the weight of the last learned tree when pruning 
condition takes place and *w_t_dart* when the algorithm behaves like dart 
(i.e., the dropped trees are put back in the ensemble at the end of the 
iteration) and *w_di* the weight of the i-th dropped tree. The normalization 
strategies are the following (TREE is the strategy used by the original DART):

- **NONE**:
    - w_t_prune = shrinkage
    - w_t_dart = shrinkage
    - w_di = w_di
- **TREE**:
    - w_t_prune = shrinkage
    - w_t_dart = shrinkage / (shrinkage + k)
    - w_di = w_di * k / (k + shrinkage)
- **TREE_ADAPTIVE**:
    - w_t_prune = shrinkage / (shrinkage + k)
    - w_t_dart = shrinkage / (shrinkage + k)
    - w_di = w_di * k / (k + shrinkage)
- **TREE_BOOST3**:
    - w_t_prune = (shrinkage * 3) / (shrinkage * 3 + k)
    - w_t_dart = (shrinkage * 3) / (shrinkage * 3 + k)
    - w_di = w_di * k / (k + shrinkage * 3)
- **WEIGHTED**: let *sum_weights* be the sum of the weights of the dropped 
trees. Then:
    - w_t_prune = shrinkage
    - w_t_dart = shrinkage / (sum_weights + shrinkage)
    - w_di = w_di * sum_weights / (sum_weights + shrinkage)
- **FOREST**:
    - w_t_prune = shrinkage
    - w_t_dart = shrinkage / (shrinkage + 1)
    - w_di = w_di * 1 / (1 + shrinkage)
- **LINESEARCH**: adopts a line search process (take a look at 
[CLEaVER](cleaver.md) readme for instructions on how it 
works and how to use it) for optimizing the weight of the last learned tree 
without considering the dropped trees. Let *opt_weight_t* be this weight. Then:
    - w_t_prune = opt_weight_t
    - w_t_dart = opt_weight_t / (opt_weight_t + k)
    - w_di = w_di * k / (k + opt_weight_t)
- **CONTR**: let contr_t the contribution to the final score of the last 
learned tree and contr_dropped the contribution to the final score of the 
dropped trees. Then: 
    - w_t_prune = contr_dropped / contr_t * shrinkage
    - w_t_dart = contr_t / (contr_dropped + contr_t)
    - w_di = w_di * contr_dropped / (contr_dropped + contr_t)
- **WCONTR**: same as *CONTR* but multiplying the contribution of each tree 
by the weight of the tree.
- **LMART_ADAPTIVE**: let *n* be the size of the ensemble at the current 
iteration and *dropout_rate* the dropout rate. Then:
    - w_t_prune = shrinkage / (dropout_rate * n + shrinkage);

Moreover, X-DART implements various strategies for setting the number *k* 
of trees to dropout at each iteration (**adaptive-type** parameter). Indeed, 
differently from DART, where *k* is set to a ratio of the ensemble size, 
X-DART support also a fixed strategy, i.e., where *k* is set to a constant 
value independent from the ensemble size, and an adaptive variant, where 
*k* is modified at each iteration dependently from the training behaviour 
of the algorithm. The adaptive strategy initially set *k=1*, then *k* is 
increased by a constant value *X* at each iteration. Whenever the loss of the
new model after the introduction of the new learned tree, either with or 
without permanent removal of the dropout set, improves over the smallest loss
observed so far, this is interpreted as the discovery of a promising search 
direction and the value of k is reset to *Y* for the subsequent iteration. 
Let *dropout_rate* be the dropout rate, then *X* and *Y* take different 
values depending from the strategy selected:

- **FIXED**: implements both the ratio and fixed strategies. If 
*dropout_rate* is < 1, then X-DART behaves like DART. Thus, at each iteration:
    - k = dropout_rate * ensemble_size
    
    Otherwise, if dropout_rate is >= 1, *k* is fixed among the iterations to:
    - k = dropout_rate
- **PLUS1_DIV2**: adaptive strategy where X=1 and Y=Y/2
- **PLUSHALF_DIV2**: adaptive strategy where X=1/2 and Y=Y/2
- **PLUSONETHIRD_DIV2**: adaptive strategy where X=1/3 and Y=Y/2
- **PLUSHALF_RESET**: adaptive strategy where X=1/2 and Y=1
- **PLUSHALF_RESET_LB1_UB5**: adaptive strategy where X=1/2 and Y=Y/2. This 
strategy set also an upper-bound limit that avoid *k* to become higher than 5.
- **PLUSHALF_RESET_LB1_UB10**: as *PLUSHALF_RESET_LB1_UB5* but using an 
upper-bound of 10.
- **PLUSHALF_RESET_LB1_UBRD**: as *PLUSHALF_RESET_LB1_UB5* but using an 
upper-bound dependent from the ensemble size (set to dropout_rate * 
ensemble_size).

Finally, the other parameters useful for running X-DART are the followings:
- **rate-drop**: dropout rate (set **k** to a fixed value or to a fraction of
 the ensemble size)
- **skip-drop**: probability of skipping the dropout phase. When set to 1 the
 training behave like the base algorithm adopted, i.e, LAMBDAMART. When set 
 to 0 the dropout phase is executed at every iteration (like the original DART).
- **keep-drop**: If set, whenever the loss of the new model after the
 introduction of the new learned tree improves over the smallest loss 
 observed so far, the dropped trees are permanently pruned from the model. 
 This is the key parameter to differentiate DART from X-DART, since DART does
 not execute any permanent pruning and thus *keep-drop* has not to be set.
- **best-on-train**: if set, the pruning condition (improvement) is 
measured on the training set, otherwise on the validation.
- **random-keep**: probability to randomly prune in a permanent way trees in 
the dropout set D independently from having observed any improvement to the loss
- **drop-on-best**: if set to true, the pruning condition (improvement) is 
measured w.r.t. the best iteration observed so far, otherwise on the last 
iteration.

Here is an example on how to use QuickRank for training a model using the 
original DART algorithm with a dropout rate set to 1.5%, as suggested in the 
article of Rashmi and Gilad-Bachrach:

```
./bin/quicklearn \
  --algo DART \
  --train quickranktestdata/msn1/msn1.fold1.train.5k.txt \
  --valid quickranktestdata/msn1/msn1.fold1.vali.5k.txt \
  --model-out dart-model.xml \
  --num-trees 100 \
  --shrinkage 0.1 \
  --sample-type UNIFORM \
  --normalize-type TREE \
  --adaptive-type FIXED \
  --rate-drop 0.015
```

Alternatively, you can train a X-DART model with the same parameter with the 
following command line:

```
./bin/quicklearn \
  --algo DART \
  --train quickranktestdata/msn1/msn1.fold1.train.5k.txt \
  --valid quickranktestdata/msn1/msn1.fold1.vali.5k.txt \
  --model-out dart-model.xml \
  --num-trees 100 \
  --shrinkage 0.1 \
  --sample-type UNIFORM \
  --normalize-type TREE \
  --adaptive-type FIXED \
  --rate-drop 0.015 \  
  --keep-drop \
  --best-on-train
```

According to the X-DART article, the best performance have been obtained 
using the UNIFORM sampling strategy for selecting the trees to dropout, the 
TREE normalization strategy for normalizing the weights (like DART) and the 
PLUSHALF_RESET_LB1_UBRD adaptive strategy for deciding which *k* to use at 
each iteration (as well using a dropout_rate=0.015) 

```
./bin/quicklearn \
  --algo DART \
  --train quickranktestdata/msn1/msn1.fold1.train.5k.txt \
  --valid quickranktestdata/msn1/msn1.fold1.vali.5k.txt \
  --model-out dart-model.xml \
  --num-trees 100 \
  --shrinkage 0.1 \
  --sample-type UNIFORM \
  --normalize-type TREE \
  --adaptive-type PLUSHALF_RESET_LB1_UBRD \
  --rate-drop 0.015 \  
  --keep-drop \
  --best-on-train
```

The examples above are done on sample data. We recommend to train DART/X-DART
on standard letor datasets (istella, msn, yahoo, etc.) for results that are 
consistent with the reference paper.

Finally, to score a DART or X-DART model (that is a standard ensemble of 
regression trees):

 ```
 ./bin/quicklearn \
   --model-in dart-model.xml \
   --test quickranktestdata/msn1/msn1.fold1.test.5k.txt \
 ```

References
-------
[1] K.V. Rashmi and R. Gilad-Bachrach. 2015. Dart: Dropouts meet multiple 
additive regression trees. Journal of Machine Learning Research 38 (2015)

[2] C. Lucchese, F. M. Nardini, S. Orlando, R. Perego, F. Silvestri, and S. 
Trani. Post- Learning Optimization of Tree Ensembles for Efficient Ranking. In 
Proceedings of ACM SIGIR 2016.

Acknowledgements
-------

If you use the **X-DART**, please acknowledge the following paper:

 - C. Lucchese, F. M. Nardini, S. Orlando, R. Perego, and S. Trani. 
 **X-DART: Blending Dropout and Pruning for Efficient Learning to Rank**.
 *ACM SIGIR Conference on Research and Development in Information Retrieval*,
  (2017).
[LINK](https://doi.org/10.1145/3077136.3080725).

License
-------
© Contributors, 2016. Licensed under an [Reciprocal Public License (RPL-1.5)](https://opensource.org/licenses/RPL-1.5).
