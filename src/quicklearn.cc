/*
 * QuickRank - A C++ suite of Learning to Rank algorithms
 * Webpage: http://quickrank.isti.cnr.it/
 * Contact: quickrank@isti.cnr.it
 *
 * Unless explicitly acquired and licensed from Licensor under another
 * license, the contents of this file are subject to the Reciprocal Public
 * License ("RPL") Version 1.5, or subsequent versions as allowed by the RPL,
 * and You may not copy or use this file in either source code or executable
 * form, except in compliance with the terms and conditions of the RPL.
 *
 * All software distributed under the RPL is provided strictly on an "AS
 * IS" basis, WITHOUT WARRANTY OF ANY KIND, EITHER EXPRESS OR IMPLIED, AND
 * LICENSOR HEREBY DISCLAIMS ALL SUCH WARRANTIES, INCLUDING WITHOUT
 * LIMITATION, ANY WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
 * PURPOSE, QUIET ENJOYMENT, OR NON-INFRINGEMENT. See the RPL for specific
 * language governing rights and limitations under the RPL.
 *
 * Contributor:
 *   HPC. Laboratory - ISTI - CNR - http://hpc.isti.cnr.it/
 */

/**
 * \mainpage QuickRank: A C++ suite of Learning to Rank algorithms
 *
 * \section nutshell QuickRank in a nutshell
 *
 * QuickRank is an efficient Learning-to-Rank (L-t-R) Toolkit providing several
 * C++ implementation of L-t-R algorithms.
 *
 * The algorithms currently implemented are:
 *   - \b GBRT: J. H. Friedman. Greedy function approximation: a gradient boosting machine.
 *   Annals of Statistics, pages 1189–1232,
 2001.
 *   - \b LamdaMART: Q. Wu, C. Burges, K. Svore, and J. Gao.
 *   Adapting boosting for information retrieval measures.
 *   Information Retrieval, 2010.
 *   - \b Oblivious \b GBRT / \b LambdaMart: Inspired to I. Segalovich. Machine learning in search quality at Yandex.
 *   Invited Talk, ACM SIGIR, 2010.
 *   - \b CoordinateAscent: Metzler, D., Croft, W.B.: Linear feature-based models for information retrieval.
 *   Information Retrieval 10(3), 257–274 (2007).
 *   - \b RankBoost: Freund, Y., Iyer, R., Schapire, R. E., & Singer, Y. An efficient boosting algorithm
 *   for combining preferences. The Journal of machine learning research, 4, 933-969 (2003).
 *
 * \subsection download Get QuickRank
 * The homepage of QuickRank is available at: <a href="http://quickrank.isti.cnr.it">http://quickrank.isti.cnr.it</a>.
 *
 * \subsection compile Compile and Use QuickRank
 *
 * - clone the GitHub repository as shown in the dedicated section of the QuickRank homepage.
 *
 * - run "make";
 *
 * - run "bin/quicklearn -h" to have the list of command line options.
 *
 * - have fun! :)
 */

/// \todo TODO: (by cla) Decide on outpuformat, logging and similar.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <time.h>
#include <iostream>
#include <iomanip>
#include <memory>
#include <stdio.h>
#include <unistd.h>
#include <fstream>

#include "paramsmap/paramsmap.h"

#include "learning/forests/mart.h"
#include "learning/forests/lambdamart.h"
#include "learning/forests/obliviousmart.h"
#include "learning/forests/obliviouslambdamart.h"
#include "learning/forests/rankboost.h"
#include "learning/linear/coordinate_ascent.h"
#include "learning/linear/line_search.h"
#include "learning/custom/custom_ltr.h"
#include "optimization/post_learning/pruning/ensemble_pruning.h"

#include "metric/ir/tndcg.h"
#include "metric/ir/ndcg.h"
#include "metric/ir/dcg.h"
#include "metric/ir/map.h"

#include "driver/driver.h"

#include "pugixml/src/pugixml.hpp"


void print_logo() {
  if (isatty(fileno(stdout))) {
    std::string color_reset = "\033[0m";
    std::string color_logo = "\033[1m\033[32m";
    std::cout << color_logo << std::endl
        << "      _____  _____" << std::endl
        << "     /    / /____/" << std::endl
        << "    /____\\ /    \\          QuickRank has been developed by hpc.isti.cnr.it" << std::endl
        << "    ::Quick:Rank::                                   quickrank@isti.cnr.it" << std::endl
        << color_reset << std::endl;
  } else {
    std::cout << std::endl
        << "      _____  _____" << std::endl
        << "     /    / /____/" << std::endl
        << "    /____\\ /    \\          QuickRank has been developed by hpc.isti.cnr.it" << std::endl
        << "    ::Quick:Rank::                                   quickrank@isti.cnr.it" << std::endl
        << std::endl;
  }
}

int main(int argc, char *argv[]) {

  print_logo();

  std::cout << std::fixed;
  srand(time(NULL));

  // default parameters
  std::string algorithm_string = quickrank::learning::forests::LambdaMart::NAME_;
  size_t ntrees = 1000;
  double shrinkage = 0.10f;
  size_t nthresholds = 0;
  size_t minleafsupport = 1;
  size_t esr = 100;
  size_t ntreeleaves = 10;
  size_t treedepth = 3;
  std::string train_metric_string = quickrank::metric::ir::Ndcg::NAME_;
  size_t train_cutoff = 10;
  std::string test_metric_string = quickrank::metric::ir::Ndcg::NAME_;
  size_t test_cutoff = 10;
  size_t partial_save = 100;
//  bool detailed_testing = false;
//  std::string opt_algo_string;
//  std::string opt_method_string;
//  bool ensemble_pruning_with_line_search = false;
//  bool adaptive = false;
//  std::string training_filename;
//  std::string validation_filename;
//  std::string test_filename;
//  std::string features_filename;
//  std::string model_filename;
//  std::string scores_filename;
//  std::string xml_filename;
//  std::string xml_linesearch_filename;
//  std::string c_filename;
//  std::string model_code_type;

  // ------------------------------------------
  // Coordinate ascent added by Chiara Pierucci
  unsigned int num_points = 21;
  unsigned int max_iterations = 100;
  double window_size = 10.0;
  double reduction_factor = 0.95;
  unsigned int max_failed_vali = 20;

  // ------------------------------------------
  // Ensemble Pruning added by Salvatore Trani
//  double epruning_rate;
//  std::string epruning_method_name;

  ParamsMap pmap;

  // Declare the supported options.
  pmap.addMessage("Training phase - general options:");
  pmap.addOptionWithArg("algo",
                        "LtR algorithm ["
                            + quickrank::learning::forests::Mart::NAME_ + "|"
                            + quickrank::learning::forests::LambdaMart::NAME_ + "|"
                            + quickrank::learning::forests::ObliviousMart::NAME_ + "|"
                            + quickrank::learning::forests::ObliviousLambdaMart::NAME_ + "|"
                            + quickrank::learning::forests::Rankboost::NAME_ + "|"
                            + quickrank::learning::linear::CoordinateAscent::NAME_ + "|"
                            + quickrank::learning::linear::LineSearch::NAME_ + "|"
                            + quickrank::learning::CustomLTR::NAME_ + "]",
                        algorithm_string);

  pmap.addOptionWithArg("train-metric",
                        "set train metric [" + quickrank::metric::ir::Dcg::NAME_ + "|"
                            + quickrank::metric::ir::Ndcg::NAME_ + "|"
                            + quickrank::metric::ir::Tndcg::NAME_ + "|"
                            + quickrank::metric::ir::Map::NAME_ + "]",
                        train_metric_string);

  pmap.addOptionWithArg("train-cutoff",
                        "set train metric cutoff",
                        train_cutoff);

  pmap.addOptionWithArg("partial",
                        "set partial file save frequency",
                        partial_save);

  pmap.addOptionWithArg<std::string>("train", "set training file");

  pmap.addOptionWithArg<std::string>("valid", "set validation file");

  pmap.addOptionWithArg<std::string>("features", "set features file");

  pmap.addOptionWithArg<std::string>("model",
                                     "set output model file for training or "
                                         "input model file for testing");


  // --------------------------------------------------------
  pmap.addMessage("Training phase - specific options for tree-based models:");
  pmap.addOptionWithArg("num-trees",
                        "set number of trees",
                        ntrees);

  pmap.addOptionWithArg("shrinkage",
                        "set shrinkage",
                        shrinkage);

  pmap.addOptionWithArg("num-thresholds",
                        "set number of thresholds",
                        nthresholds);

  pmap.addOptionWithArg("min-leaf-support",
                        "set minimum number of leaf support",
                        minleafsupport);

  pmap.addOptionWithArg("end-after-rounds",
                        "set num. rounds with no boost in validation before ending (if 0 disabled)",
                        esr);

  pmap.addOptionWithArg("num-leaves",
                        "set number of leaves [applies only to Mart/LambdaMart]",
                        ntreeleaves);

  pmap.addOptionWithArg("tree-depth",
                        "set tree depth [applies only to Oblivious Mart/LambdaMart]",
                        treedepth);


  // --------------------------------------------------------
  // CoordinateAscent and LineSearch options
  // add by Chiara Pierucci and Salvatore Trani
  pmap.addMessage("Training phase - specific options for Coordinate Ascent and Line Search:");
  pmap.addOptionWithArg("num-samples",
                        "set number of samples in search window",
                        num_points);

  pmap.addOptionWithArg("window-size",
                        "set search window size",
                        window_size);

  pmap.addOptionWithArg("reduction-factor",
                        "set window reduction factor",
                        reduction_factor);

  pmap.addOptionWithArg("max-iterations",
                        "set number of max iterations",
                        max_iterations);

  pmap.addOptionWithArg("max-failed-valid",
                        "set number of fails on validation before exit",
                        max_failed_vali);


  // --------------------------------------------------------
  // LineSearch extra options add by Salvatore Trani
  pmap.addMessage("Training phase - specific options for Line Search:");
  pmap.addOption("adaptive",
                 "enable adaptive reduction factor (based on last iteration "
                     "metric gain)");
  pmap.addOptionWithArg<std::string>("train-partial",
                                     "set training file with partial scores "
                                         "(input for loading or output for "
                                         "saving)");
  pmap.addOptionWithArg<std::string>("valid-partial",
                                     "set validation file with partial scores "
                                         "(input for loading or output for "
                                         "saving)");


  // --------------------------------------------------------
  // Optimization options add by Salvatore Trani
  pmap.addMessage("Optimization phase - general options:");
  pmap.addOptionWithArg<std::string>(
      "opt-algo",
      "Optimization alghoritm [" +
          quickrank::optimization::post_learning::pruning::EnsemblePruning::NAME_
          + "]");

  std::string pruningMethods = "";
  for (auto i: quickrank::optimization::post_learning::pruning::EnsemblePruning::pruningMethodNames) {
    pruningMethods += i + "|";
  }
  pruningMethods = pruningMethods.substr(0, pruningMethods.size() - 1);

  pmap.addOptionWithArg<std::string>(
      "opt-method",
      "Optimization method: " +
          quickrank::optimization::post_learning::pruning::EnsemblePruning::NAME_
          + " [" + pruningMethods + "]");

  pmap.addOptionWithArg<std::string>(
      "opt-model",
      "set output model file for optimization or input model file for testing");

  pmap.addOptionWithArg<std::string>(
      "opt-algo-model",
      "set output algorithm model file post optimization");


  // --------------------------------------------------------
  pmap.addMessage("Optimization phase - specific options for ensemble pruning:");
  pmap.addOptionWithArg<double>("pruning-rate",
                                "ensemble to prune (either as a ratio with "
                                    "respect to ensemble size or as an absolute "
                                    "number of estimators to prune)");

  pmap.addOption("with-line-search",
                 "ensemble pruning is made in conjunction with line search "
                     "[related parameters accepted]");

  pmap.addOptionWithArg<std::string>("line-search-model",
                                     "set line search XML file path for "
                                         "loading line search model (options "
                                         "and already trained weights)");


  // --------------------------------------------------------
  pmap.addMessage("Test phase - general options:");
  pmap.addOptionWithArg("test-metric",
                        "set test metric [" + quickrank::metric::ir::Dcg::NAME_ + "|"
                            + quickrank::metric::ir::Ndcg::NAME_ + "|"
                            + quickrank::metric::ir::Tndcg::NAME_ + "|"
                            + quickrank::metric::ir::Map::NAME_ + "]",
                        test_metric_string);

  pmap.addOptionWithArg("test-cutoff",
                        "set test metric cutoff",
                        test_cutoff);

  pmap.addOptionWithArg<std::string>("test", "set testing file");

  pmap.addOptionWithArg<std::string>("scores", "set output scores file");

  pmap.addOption("detailed",
                 "enable detailed testing [applies only to ensemble models]");


  // --------------------------------------------------------
  pmap.addMessage("Code generation - general options:");
  pmap.addOptionWithArg<std::string>("dump-model", "set XML model file path");

  pmap.addOptionWithArg<std::string>("dump-code", "set C code file path");

  pmap.addOptionWithArg("dump-type",
                        "set C code generation strategy. Allowed options are:"
                            " \"baseline\", \"oblivious\". \"vpred\".",
                        std::string("baseline"));






  // --------------------------------------------------------
  pmap.addMessage("Help options:");
  pmap.addOption("help", "h", "print help message");


  bool parse_status = pmap.parse(argc, argv);
  if (!parse_status || pmap.isSet("help")) {
    std::cout << pmap.help();
    return 1;
  }

  return quickrank::driver::Driver::run(pmap);
}
