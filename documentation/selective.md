Selective Gradient Boosting
==========

Selective Gradient Boosting is a new step-wise algorithm introducing a tunable 
and dynamic selection of negative instances within λ-Mart. Similarly to λ-Mart
it produces an ensemble of binary decision trees but performs at training 
time a dynamic selection of the negative examples to be kept in the training 
set. In particular, the algorithm selects the top-scored negative instances 
within the lists associated with each query with the aim of minimizing the 
mis-ranking risk (this idea originate in [1], where authors 
preliminarly demonstrated the validity of a rank-based sampling strategy). Due
to the characteristics of the NDCG metric used to evaluate the quality of 
the learned model, we need to discriminate the few positive instances that 
must be pushed in the top positions of the scored lists, from the plenty of 
negative instances in the training set. Indeed top-scored negative instances 
are exactly those being more likely to be ranked above relevant instances, 
thus severely hindering the ranking quality.

Unlike other sampling methods proposed in the literature, our method does not
simply aim at sampling the training set to reduce the training time without 
affecting the effectiveness of the trained model. Conversely, the proposed 
method is able to dynamically choose the “most informative” negative examples
of the training set, so as to improve the final effectiveness of the learned
model.

Here is an example on how to use QuickRank for training a model using the 
Selective Gradient Boosting algorithm with a sampling frequency of 1 and a 
sampling rate set to 5% (this parameter highlights the number of irrelevant 
instances to be selected among the top ranked ones):

```
./bin/quicklearn \
  --algo LAMBDAMART-SELECTIVE \
  --train quickranktestdata/msn1/msn1.fold1.train.5k.txt \
  --valid quickranktestdata/msn1/msn1.fold1.vali.5k.txt \
  --model-out selective-model.xml \
  --num-trees 100 \
  --num-leaves 32 \
  --shrinkage 0.1 \
  --sampling-iterations 1 \
  --rank-sampling-factor 0.05
```

According to the Selective Gradient Boosting article, the best performance have 
been obtained performing the sampling at each iteration (sampling frequency 
of 1) and adopting a sampling rate of 1% for a dataset highly unbalanced. For
 traditional dataset less aggressive sampling rates should be adopted.

```
./bin/quicklearn \
  --algo LAMBDAMART-SELECTIVE \
  --train quickranktestdata/msn1/msn1.fold1.train.5k.txt \
  --valid quickranktestdata/msn1/msn1.fold1.vali.5k.txt \
  --model-out selective-model.xml \
  --num-trees 1000 \
  --num-leaves 64 \
  --shrinkage 0.05 \
  --sampling-iterations 1 \
  --rank-sampling-factor 0.01
```

The examples above are done on sample data. We recommend to train the algorithm
on standard letor datasets (istella, msn, yahoo, etc.) for results that are 
consistent with the reference paper. The dataset adopted for demonstrating 
the validity of the proposed algorithm in the original article can be found 
here: [Istella-X](http://quickrank.isti.cnr.it/istella-dataset/)

Finally, to score the trained model (that is a standard ensemble of 
regression trees):

 ```
 ./bin/quicklearn \
   --model-in selective-model.xml \
   --test quickranktestdata/msn1/msn1.fold1.test.5k.txt \
 ```

References
-------
[1] C. Lucchese, F. M. Nardini, R. Perego, and S. Trani. 
The Impact of Negative Samples on Learning to Rank. In Proceedings of ACM 
ICTIR 2017.

Acknowledgements
-------

If you use the **Selective Gradient Boosting** algorithm, please acknowledge 
the following paper:

 - C. Lucchese, F. M. Nardini, S. Orlando, R. Perego, and S. Trani. 
 **Selective Gradient Boosting for Effective Learning to Rank**.
 *ACM SIGIR Conference on Research and Development in Information Retrieval*,
  (2018).
[LINK](https://doi.org/10.1145/3209978.3210048).

License
-------
© Contributors, 2016. Licensed under an [Reciprocal Public License (RPL-1.5)](https://opensource.org/licenses/RPL-1.5).
