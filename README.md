<img src=https://raw.githubusercontent.com/hpclab/quickrank/master/api-documentation/banner.png>

QuickRank: A C++ suite of Learning-to-Rank algorithms
===========

QuickRank is an efficient Learning-to-Rank toolkit providing several C++ implementation of LtR algorithms. QuickRank was designed and developed with efficiency in mind.

The LtR algorithms currently implemented are:

 - **GBRT**: J. H. Friedman. *Greedy function approximation: a gradient boosting machine*. Annals of Statistics, pages 1189–1232, 2001.
 - **LamdaMART**: Q. Wu, C. Burges, K. Svore, and J. Gao. *Adapting boosting for information retrieval measures*. Information Retrieval, 2010.
 - **Oblivious GBRT / LamdaMART**: Inspired to I. Segalovich. *Machine learning in search quality at yandex*. Invited Talk, ACM SIGIR, 2010.
 - **CoordinateAscent**: Metzler, D., Croft, W.B.. *Linear feature-based models for information retrieval*. Information Retrieval 10(3), pages 257–274, 2007.
 - **LineSearch**: D. G. Luenberger. *Linear and nonlinear programming*. Addison Wesley, 1984.
 - **RankBoost**: Freund, Y., Iyer, R., Schapire, R. E., & Singer, Y. *An efficient boosting algorithm for combining preferences*. JMLR, 4, 933-969 (2003).
 - **DART**: K.V. Rashmi and R. Gilad-Bachrach. *Dart: Dropouts meet multiple 
 additive regression trees*. JMLR, 38 (2015).
 - **Selective**:  C. Lucchese, F. M. Nardini, S. Orlando, R. Perego and S. 
 Trani. *Selective Gradient Boosting for Effective Learning to Rank*. ACM 
 SIGIR, 2018. [README](documentation/selective.md)

QuickRank also provides novel learning optimizations. Currently implemented optimizers are:

  - **CLEAVER**: C. Lucchese, F. M. Nardini, S. Orlando, R. Perego, F. Silvestri, S. Trani. *Post-Learning Optimization of Tree Ensembles for Efficient Ranking*. In Proc. ACM SIGIR, 2016. [README](documentation/cleaver.md)
  - **X-CLEAVER**: C. Lucchese, F. M. Nardini, S. Orlando, R. Perego, F. Silvestri, S. Trani. *X-CLEaVER: Learning Ranking Ensembles by Growing and Pruning Trees*. Paper under revision.
  - **X-DART**: C. Lucchese, F. M. Nardini, S. Orlando, R. Perego and S. Trani. *X-DART: Blending Dropout and Pruning for Efficient Learning to Rank*. In Proc. ACM SIGIR, 2017. [README](documentation/xdart.md)

How to build
-------

QuickRank needs [gcc](https://gcc.gnu.org/) 4.9 (or above), [CMake](http://www.cmake.org/) 2.8 (or above) and [git](https://git-scm.com/). Follow the instructions below to install the required tools.

On Mac OS X (HomeBrew and XCode command line tools are required):

	xcode-select --install;
	brew install gcc cmake git;

On Ubuntu Linux:

	sudo apt-get update;
	sudo apt-get install gcc-5 cmake git;

Once the compiler and the build system are installed, you can get the latest stable source packages (including the dependencies) using git:

	git clone --recursive https://github.com/hpclab/quickrank.git

QuickRank relies on the CMake build system. The root directory contains CMake files that can be used to build the project from the command line or using an IDE which supports CMake. The instructions which follows describe how to build the project from the command line, but the same actions can be performed using appropriate GUI tools. 

Create a temporary build folder and change your working directory to it:

	cd quickrank
	mkdir build_
	cd build_

The Makefile generator can build the project in only one configuration, so you need to build a separate folder for each configuration. As an example, to use the Release configuration (which is by default, Debug is similar) you need to execute (take care of the compiler paths):

```
cmake .. \
-DCMAKE_CXX_COMPILER=/usr/local/bin/g++-5 \
-DCMAKE_BUILD_TYPE=Release
```
Finally to compile Quickrank:

	make

To compile tests:

	make unit-tests

And wait for the compilation to finish. The result will be the QuickRank executable placed in the bin directory of the project root.

If you would like to execute the unit-tests, you need to run in the root directory:

	./bin/unit-test

How to use
-------

Running the QuickRank binary with the `-h` option, it shows the help:

```
./bin/quicklearn -h

      _____  _____
     /    / /____/
    /____\ /    \          QuickRank has been developed by hpc.isti.cnr.it
    ::Quick:Rank::                                   quickrank@isti.cnr.it


Training phase - general options:
  --algo <arg> (LAMBDAMART)             LtR algorithm:
                                        [MART|LAMBDAMART|OBVMART|OBVLAMBDAMART|DART|
                                        RANKBOOST|COORDASC|LINESEARCH|CUSTOM].
  --train-metric <arg> (NDCG)           set train metric: [DCG|NDCG|TNDCG|MAP].
  --train-cutoff <arg> (10)             set train metric cutoff.
  --partial <arg> (100)                 set partial file save frequency.
  --train <arg>                         set training file.
  --valid <arg>                         set validation file.
  --features <arg>                      set features file.
  --model-in <arg>                      set input model file
                                        (for testing, re-training or optimization)
  --model-out <arg>                     set output model file
  --skip-train                          skip training phase.
  --restart-train                       restart training phase from a previous trained model.

Training phase - specific options for tree-based models:
  --num-trees <arg> (1000)              set number of trees.
  --shrinkage <arg> (0.1)               set shrinkage.
  --num-thresholds <arg> (0)            set number of thresholds.
  --min-leaf-support <arg> (1)          set minimum number of leaf support.
  --end-after-rounds <arg> (100)        set num. rounds with no gain in validation
                                        before ending (if 0 disabled).
  --num-leaves <arg> (10)               set number of leaves
                                        [applies only to MART/LambdaMART].
  --tree-depth <arg> (3)                set tree depth
                                        [applies only to ObliviousMART/ObliviousLambdaMART].

Training phase - specific options for Meta LtR models:
  --meta-algo <arg>                     Meta LtR algorithm:
                                        [METACLEAVER].
  --final-num-trees <arg>               set number of final trees.
  --opt-last-only                       optimization executed only on trees learned
                                        in last iteration.
  --meta-end-after-rounds <arg>         set num. rounds with no gain in validation
                                        before ending (if 0 disabled) on meta LtR models.
  --meta-verbose                        Increase verbosity of Meta Algorithm,
                                        showing each step in detail.

Training phase - specific options for Dart:
  --sample-type <arg> (UNIFORM)         sampling type of trees. [UNIFORM|WEIGHTED|WEIGHTED_INV|COUNT2|COUNT3|COUNT2N|COUNT3N|TOP_FIFTY|CONTR|CONTR_INV|WCONTR|WCONTR_INV|TOP_WCONTR|LESS_WCONTR].
  --normalize-type <arg> (TREE)         normalization type of trees. [TREE|NONE|WEIGHTED|FOREST|TREE_ADAPTIVE|LINESEARCH|TREE_BOOST3|CONTR|WCONTR|LMART_ADAPTIVE|LMART_ADAPTIVE_SIZE].
  --adaptive-type <arg> (FIXED)         adaptive type for choosing number of trees to dropout:
                                        [FIXED|PLUS1_DIV2|PLUSHALF_DIV2|PLUSONETHIRD_DIV2|PLUSHALF_RESET|PLUSHALF_RESET_LB1_UB5|PLUSHALF_RESET_LB1_UB10|PLUSHALF_RESET_LB1_UBRD].
  --rate-drop <arg> (0.1)               set dropout rate
  --skip-drop <arg> (0)                 set probability of skipping dropout
  --keep-drop                           keep the dropped trees out of the ensembleif the performance of the model improved
  --best-on-train                       Calculate the best performance on training (o/w valid)
  --random-keep <arg> (0)               keep the dropped trees out of the ensemble
                                        for every drop
  --drop-on-best                        Perform the drop-out based on best perfomance (o/w last)

Training phase - specific options for Coordinate Ascent and Line Search:
  --num-samples <arg> (21)              set number of samples in search window.
  --window-size <arg> (10)              set search window size.
  --reduction-factor <arg> (0.95)       set window reduction factor.
  --max-iterations <arg> (100)          set number of max iterations.
  --max-failed-valid <arg> (20)         set number of fails on validation before exit.

Training phase - specific options for Line Search:
  --adaptive                            enable adaptive reduction factor
                                        (based on last iteration metric gain).
  --train-partial <arg>                 set training file with partial scores
                                        (input for loading or output for saving).
  --valid-partial <arg>                 set validation file with partial scores
                                        (input for loading or output for saving).

Optimization phase - general options:
  --opt-algo <arg>                      Optimization algorithm: [CLEAVER].
  --opt-method <arg>                    Optimization method: CLEAVER
                                        [RANDOM|RANDOM_ADV|LOW_WEIGHTS|SKIP|LAST|QUALITY_LOSS|QUALITY_LOSS_ADV|SCORE_LOSS].
  --opt-model <arg>                     set output model file for optimization
                                        or input model file for testing.
  --opt-algo-model <arg>                set output algorithm model file post optimization.

Optimization phase - specific options for ensemble pruning:
  --pruning-rate <arg>                  ensemble to prune (either as a ratio with
                                        respect to ensemble size or as an absolute
                                        number of estimators to prune).
  --with-line-search                    ensemble pruning is made in conjunction
                                        with line search [related parameters accepted].
  --line-search-model <arg>             set line search XML file path for
                                        loading line search model (options
                                        and already trained weights).

Test phase - general options:
  --test-metric <arg> (NDCG)            set test metric: [DCG|NDCG|TNDCG|RMSE|MAP].
  --test-cutoff <arg> (10)              set test metric cutoff.
  --test <arg>                          set testing file.
  --scores <arg>                        set output scores file.
  --detailed                            enable detailed testing [applies only to ensemble models].

Code generation - general options:
  --model-file <arg>                    set XML model file path.
  --code-file <arg>                     set C code file path.
  --generator <arg> (condop)            set C code generation strategy. Allowed options are:
                                        -  "condop" (conditional operators),
                                        -  "oblivious" (optimized code for oblivious trees),
                                        -  "vpred" (intermediate code used by VPRED).

Help options:
  -h,--help                             print help message.
```

Some parameters have a default value (described in round brackets), which means you can skip their assignation if the default value is ok for you.

### Training
Training a model with QuickRank is straightforward. Looking at the training options, you need to specify the learning algorithm, the training dataset (optionally also the validation dataset) and the metric to use. If you would like to save the trained model on a file, you need to pass also the `model-out` parameter:
```
./bin/quicklearn \
  --algo LAMBDAMART \
  --train quickranktestdata/msn1/msn1.fold1.train.5k.txt \
  --valid quickranktestdata/msn1/msn1.fold1.vali.5k.txt \
  --train-metric NDCG \
  --train-cutoff 10 \ 
  --model-out lambdamart-model.xml
```

Depending from the learning algorithm adopted, you have to pass additional parameters, e.g., the number of trees for ensemble-based models (and many others). Some parameters have a default value, which means if you can skip them they will use that value for training.

### Testing

To test a model, you could specify the test option in the previous command, or load a previously saved model. The predicted scores can be saved on a file (one score per row, preserving the order of the test dataset). 

```
./bin/quicklearn \
  --model-in lambdamart-model.xml \
  --test quickranktestdata/msn1/msn1.fold1.test.5k.txt \
  --test-metric NDCG \
  --test-cutoff 10 \ 
  --scores scores.txt
```

With the ```--detailed``` option, valid only for ensemble-based algorithms, QuickRank will save in a SVM-light format (which consequently can be used as input dataset for other learning algorithms) the partial scores given by each weak ranker to the prediction of the documents (one row per document, a feature for each ensemble, preserving the order of the ensembles in the model and of the documents in the dataset).


### Efficient Scoring

QuickRank can translate learnt tree-based models into efficient C++ source code that can be used to score documents efficiently. See a more detailed description [here](documentation/quickscore.md).

### Optimization

QuickRank introduces the concept of optimizers, i.e., algorithms than are 
executed before or after the training phase is executed. An optimizer could process either the dataset or the model, depending from its definition. Currently in QuickRank there is a single optimizer which acts in post learning by pruning an ensemble model, improving consequently its efficiency, without hindering its effectiveness.

The optimizer can be executed in pipeline with the training phase by setting the corresponding options, or as a standalone process which works on an previously trained model (or dataset). 

```
./bin/quicklearn \
  --model-in lambdamart-model.xml \
  --train quickranktestdata/msn1/msn1.fold1.train.5k.txt \
  --valid quickranktestdata/msn1/msn1.fold1.vali.5k.txt \
  --opt-algo CLEAVER \
  --opt-method QUALITY_LOSS \
  --opt-model optmization-model.xml \
  --opt-algo-model optimized-model.xml \
  --pruning-rate 0.5 \
  --with-line-search \
  --num-samples 10 \
  --window-size 1 \
  --reduction-factor 0.95 \
  --max-iterations 100 \
  --max-failed-valid 20
```

See a more detailed description [here](documentation/cleaver.md).

### Test Data

If you need a small dataset o which to test QuickRank, all you need to do is to run the following command from the ```_build``` directory mentioned the in "How to Build" section.

    make quickranktestdata

This command will clone a repository with a small sample (5k rows for each split) of the MSN1 LETOR dataset. The sample will be placed inside the directory ```./quickranktestdata/msn1/``` already in the train/test/validation split.


File Format (train/test/validation)
-------

The file format of the training, test and validation files is the same as for SVM-Light (http://svmlight.joachims.org/). This is also the format used in the LETOR datasets. Each of the following lines represents one training example and is of the following format:

	<line> .=. <target> qid:<qid> <feature>:<value> <feature>:<value> ... <feature>:<value> # <info>
	<target> .=. <float>
	<qid> .=. <positive integer>
	<feature> .=. <positive integer>
	<value> .=. <float>
	<info> .=. <string>

The target value and each of the feature/value pairs are separated by a space character. Feature/value pairs MUST be ordered by increasing feature number. Features with value zero can be skipped. The string <info> can be used to pass additional information to the kernel (e.g. non feature vector data).

Here's an example: (taken from the SVM-Rank website). Note that everything after "#" are discarded.

	3 qid:1 1:1 2:1 3:0 4:0.2 5:0 # 1A
	2 qid:1 1:0 2:0 3:1 4:0.1 5:1 # 1B 
	1 qid:1 1:0 2:1 3:0 4:0.4 5:0 # 1C
	1 qid:1 1:0 2:0 3:1 4:0.3 5:0 # 1D  
	1 qid:2 1:0 2:0 3:1 4:0.2 5:0 # 2A  
	2 qid:2 1:1 2:0 3:1 4:0.4 5:0 # 2B 
	1 qid:2 1:0 2:0 3:1 4:0.1 5:0 # 2C 
	1 qid:2 1:0 2:0 3:1 4:0.2 5:0 # 2D


Documentation
-------

Check out further [information](http://quickrank.isti.cnr.it) and [code documentation](http://quickrank.isti.cnr.it/doxygen/index.html).  

Acknowledgements
-------

If you use QuickRank, please acknowledge the following paper:
 - Capannini, G., Lucchese, C., Nardini, F. M., Orlando, S., Perego, R., and Tonellotto, N.
 **Quality versus efficiency in document scoring with learning-to-rank models.**
 *Information Processing & Management* (2016).
 [LINK](http://dx.doi.org/10.1016/j.ipm.2016.05.004).

If you use **CLEAVER**, please acknowledge the following paper:
 - C. Lucchese, F. M. Nardini, S. Orlando, R. Perego, F. Silvestri, S. Trani.
 **Post-Learning Optimization of Tree Ensembles for Efficient Ranking**.
 *ACM SIGIR Conference on Research and Development in Information Retrieval*, (2016).
 [LINK](http://dx.doi.org/10.1145/2911451.2914763).
 
If you use **X-DART**, please acknowledge the following paper:

 - C. Lucchese, F. M. Nardini, S. Orlando, R. Perego, and S. Trani. 
 **X-DART: Blending Dropout and Pruning for Efficient Learning to Rank**.
 *ACM SIGIR Conference on Research and Development in Information Retrieval*,
  (2017).
[LINK](https://doi.org/10.1145/3077136.3080725).


We will be happy to know that you are using QuickRank and to acknowledge you.
[Check the list of works using QuickRank](documentation/usedby.md)

Tools and Libraries
-------

We thank the developers of the following tools and libraries:
 - [Catch](https://github.com/philsquared/Catch): C++ Automated Test Cases in Headers
 - [PugiXML](https://github.com/zeux/pugixml): Lightweight C++ XML processing library
 - [ParamsMap](http://git.hpc.isti.cnr.it/quickrank/ParamsMap): Simple command line parser adopting easily accessible program options

License
-------
© Contributors, 2016. Licensed under an [Reciprocal Public License (RPL-1.5)](https://opensource.org/licenses/RPL-1.5).
