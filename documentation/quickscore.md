Efficient Scoring
==========

QuickRank can translate learnt tree-based models into efficient C++ source code that can be used to score documents efficiently.

This is achieved with a two step process. During the first step, a previously learnt model is translated into a C++ source.
QuickRanks implements three translation strategies:
 - `condop`: translates into nested C conditional operators of the kind `(x[feature]<=threshold) ? ( ...left...) : (...right...)`.
 - `vpred`: uses the strategy described in Asadi et al [1].
 - `oblivious`: an optimized strategies for oblivious regression trees [2].

The optimized source code is generated as follows:

    ./bin/quicklearn --model-file model.xml \
                     --code-file model.cc \
                     --generator condop

After the source code was generated it is possible to test its efficiency.
First you need to replace the file `src/scoring/ranker.cc` with the source file `model.cc` generated previously.
Then you can compile it by invoking `make quickscore` in your build directory.
Upon termination a new binary is compiled `bin/quickscore` implementing the original model.

    ./bin/quickscore  -r 10 -d dataset.test

The result shows the time need by the model to score the document in the input dataset averaged over 10 rounds.

```
      _____  _____
     /    / /____/
    /____\ /    \          QuickRank has been developed by hpc.isti.cnr.it
    ::Quick:Rank::                                   quickrank@isti.cnr.it

#	 Dataset size: 5000 x 136 (instances x features)
#	 Num queries: 43 | Avg. len: 116
       Total scoring time: 0.000139 s.
Avg. Dataset scoring time: 1.39e-05 s.
Avg.    Doc. scoring time: 2.78e-09 s.
```


[1] Asadi N, Lin J, De Vries AP.
    **Runtime optimizations for tree-based machine learning models**.
    *IEEE Transactions on Knowledge and Data Engineering*. 2014.
    [LINK](http://dx.doi.org/10.1109/TKDE.2013.73)

[2] Capannini, G., Lucchese, C., Nardini, F. M., Orlando, S., Perego, R., and Tonellotto, N.
       **Quality versus efficiency in document scoring with learning-to-rank models.**
       *Information Processing & Management* (2016).
       [LINK](http://dx.doi.org/10.1016/j.ipm.2016.05.004).