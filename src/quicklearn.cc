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
 * Please refer to <a href="https://github.com/hpclab/quickrank">https://github.com/hpclab/quickrank</a>
 * for information about compiling, using, and acknowledging QuickRank.
 */

/// \todo TODO: (by cla) Decide on outpuformat, logging and similar.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <time.h>
#include <iostream>
#include <iomanip>
#include <memory>
#include <unistd.h>
#include <fstream>
#include <metric/ir/rmse.h>

#include "paramsmap/paramsmap.h"

#include "learning/forests/mart.h"
#include "learning/forests/dart.h"
#include "learning/forests/lambdamart.h"
#include "learning/forests/obliviousmart.h"
#include "learning/forests/obliviouslambdamart.h"
#include "learning/forests/rankboost.h"
#include "learning/linear/coordinate_ascent.h"
#include "learning/linear/line_search.h"
#include "learning/custom/custom_ltr.h"
#include "learning/meta/meta_cleaver.h"
#include "optimization/post_learning/cleaver/cleaver.h"

#include "metric/ir/tndcg.h"
#include "metric/ir/map.h"
#include "metric/ir/rmse.h"

#include "driver/driver.h"

void print_logo() {
  if (isatty(fileno(stdout))) {
    std::string color_reset = "\033[0m";
    std::string color_logo = "\033[1m\033[32m";
    std::cout << color_logo << std::endl
              << "      _____  _____" << std::endl
              << "     /    / /____/" << std::endl
              << "    /____\\ /    \\        QuickRank has been developed by hpc.isti.cnr.it"
              << std::endl
              << "    ::Quick:Rank::                             mail: quickrank@isti.cnr.it"
              << std::endl
              << color_reset << std::endl;
  } else {
    std::cout << std::endl
              << "      _____  _____" << std::endl
              << "     /    / /____/" << std::endl
              << "    /____\\ /    \\          QuickRank has been developed by hpc.isti.cnr.it"
              << std::endl
              << "    ::Quick:Rank::                                   quickrank@isti.cnr.it"
              << std::endl
              << std::endl;
  }
}

int main(int argc, char *argv[]) {

  print_logo();

  std::cout << std::fixed;
  srand(time(NULL));

  // default parameters
  std::string
      algorithm_string = quickrank::learning::forests::LambdaMart::NAME_;
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

  std::string sample_type =
      quickrank::learning::forests::Dart::get_sampling_type(
          quickrank::learning::forests::Dart::SamplingType::UNIFORM);
  std::string normalize_type
      = quickrank::learning::forests::Dart::get_normalization_type(
          quickrank::learning::forests::Dart::NormalizationType::TREE);
  std::string adaptive_type
      = quickrank::learning::forests::Dart::get_adaptive_type(
          quickrank::learning::forests::Dart::AdaptiveType::FIXED);
  double rate_drop = 0.1;
  double skip_drop = 0;
  double random_keep = 0;

  // ------------------------------------------
  // Coordinate ascent added by Chiara Pierucci
  unsigned int num_points = 21;
  unsigned int max_iterations = 100;
  double window_size = 10.0;
  double reduction_factor = 0.95;
  unsigned int max_failed_vali = 20;

  ParamsMap pmap;

  // Declare the supported options.
  pmap.addMessage({"Training phase - general options:"});

  pmap.addOptionWithArg("algo", {"LtR algorithm:", "["
      + quickrank::learning::forests::Mart::NAME_
      + "|"
      + quickrank::learning::forests::LambdaMart::NAME_
      + "|"
      + quickrank::learning::forests::ObliviousMart::NAME_
      + "|"
      + quickrank::learning::forests::ObliviousLambdaMart::NAME_
      + "|"
      + quickrank::learning::forests::Dart::NAME_
      + "|", ""
      + quickrank::learning::forests::Rankboost::NAME_
      + "|"
      + quickrank::learning::linear::CoordinateAscent::NAME_
      + "|"
      + quickrank::learning::linear::LineSearch::NAME_
      + "|"
      + quickrank::learning::CustomLTR::NAME_
      + "]."}, algorithm_string);

  pmap.addOptionWithArg("train-metric",
                        {"set train metric: ["
                             + quickrank::metric::ir::Dcg::NAME_
                             + "|"
                             + quickrank::metric::ir::Ndcg::NAME_
                             + "|"
                             + quickrank::metric::ir::Tndcg::NAME_
                             + "|"
                             + quickrank::metric::ir::Map::NAME_
                             + "]."},
                        train_metric_string);

  pmap.addOptionWithArg("train-cutoff",
                        {"set train metric cutoff."},
                        train_cutoff);

  pmap.addOptionWithArg("partial",
                        {"set partial file save frequency."},
                        partial_save);

  pmap.addOptionWithArg<std::string>("train", {"set training file."});

  pmap.addOptionWithArg<std::string>("valid", {"set validation file."});

  pmap.addOptionWithArg<std::string>("features", {"set features file."});

  pmap.addOptionWithArg<std::string>("model-in",
                                     {"set input model file",
                                     "(for testing, re-training or optimization)"});
  pmap.addOptionWithArg<std::string>("model-out",
                                     {"set output model file"});
  pmap.addOption("skip-train", {"skip training phase."});
  pmap.addOption("restart-train", {"restart training phase from a previous "
                                       "trained model."});


  // --------------------------------------------------------
  pmap.addMessage({"Training phase - specific options for tree-based models:"});

  pmap.addOptionWithArg("num-trees", {"set number of trees."}, ntrees);

  pmap.addOptionWithArg("shrinkage", {"set shrinkage."}, shrinkage);

  pmap.addOptionWithArg("num-thresholds", {"set number of thresholds."},
                        nthresholds);

  pmap.addOptionWithArg("min-leaf-support",
                        {"set minimum number of leaf support."},
                        minleafsupport);

  pmap.addOptionWithArg("end-after-rounds",
                        {"set num. rounds with no gain in validation",
                         "before ending (if 0 disabled)."},
                        esr);

  pmap.addOptionWithArg("num-leaves",
                        {"set number of leaves",
                         "[applies only to MART/LambdaMART]."},
                        ntreeleaves);

  pmap.addOptionWithArg("tree-depth",
                        {"set tree depth",
                         "[applies only to ObliviousMART/ObliviousLambdaMART]."},
                        treedepth);

// --------------------------------------------------------
  pmap.addMessage({"Training phase - specific options for Meta LtR models:"});
  pmap.addOptionWithArg<std::string>("meta-algo", {"Meta LtR algorithm:", "["
      + quickrank::learning::meta::MetaCleaver::NAME_
      + "]."});
  pmap.addOptionWithArg<size_t>("final-num-trees",
                        {"set number of final trees."});
  pmap.addOption("opt-last-only",
                        {"optimization executed only on trees learned",
                         "in last iteration."});
  pmap.addOptionWithArg<size_t>("meta-end-after-rounds",
                                {"set num. rounds with no gain in validation",
                                 "before ending (if 0 disabled) on meta LtR "
                                     "models."});
  pmap.addOption("meta-verbose",
                 {"Increase verbosity of Meta Algorithm,",
                  "showing each step in detail."});

  // --------------------------------------------------------
  // Dart options add by Salvatore Trani
  pmap.addMessage(
      {"Training phase - specific options for Dart:"});

  std::string samplingMethods = "";
  for (auto i: quickrank::learning::forests::Dart::samplingTypesNames) {
    samplingMethods += i + "|";
  }
  samplingMethods = samplingMethods.substr(0, samplingMethods.size() - 1);

  pmap.addOptionWithArg("sample-type",
                        {"sampling type of trees. [" + samplingMethods + "]."},
                        sample_type);

  std::string normalizationMethods = "";
  for (auto i: quickrank::learning::forests::Dart::normalizationTypesNames) {
    normalizationMethods += i + "|";
  }
  normalizationMethods = normalizationMethods.substr(0, normalizationMethods.size() - 1);

  pmap.addOptionWithArg("normalize-type",
                        {"normalization type of trees. "
                             "[" + normalizationMethods + "]."},
                        normalize_type);

  std::string adaptiveMethods = "";
  for (auto i: quickrank::learning::forests::Dart::adaptiveTypeNames) {
    adaptiveMethods += i + "|";
  }
  adaptiveMethods = adaptiveMethods.substr(0, adaptiveMethods.size() - 1);

  pmap.addOptionWithArg("adaptive-type",
                        {"adaptive type for choosing number of trees to dropout:",
                             "[" + adaptiveMethods + "]."},
                        adaptive_type);

  pmap.addOptionWithArg("rate-drop",
                        {"set dropout rate"},
                        rate_drop);

  pmap.addOptionWithArg("skip-drop",
                        {"set probability of skipping dropout"},
                        skip_drop);

  pmap.addOption("keep-drop",
                {"keep the dropped trees out of the ensemble "
                 "if the performance of the model improved"});

  pmap.addOption("best-on-train",
                 {"Calculate the best performance on training (o/w valid)"});

  pmap.addOptionWithArg("random-keep",
                        {"keep the dropped trees out of the ensemble",
                         "for every drop"},
                        random_keep);

  pmap.addOption("drop-on-best",
                 {"Perform the drop-out based on best perfomance (o/w last)"});

  // --------------------------------------------------------
  // CoordinateAscent and LineSearch options
  // add by Chiara Pierucci and Salvatore Trani
  pmap.addMessage(
      {"Training phase - specific options for Coordinate Ascent and Line Search:"});
  pmap.addOptionWithArg("num-samples",
                        {"set number of samples in search window."},
                        num_points);

  pmap.addOptionWithArg("window-size",
                        {"set search window size."},
                        window_size);

  pmap.addOptionWithArg("reduction-factor",
                        {"set window reduction factor."},
                        reduction_factor);

  pmap.addOptionWithArg("max-iterations",
                        {"set number of max iterations."},
                        max_iterations);

  pmap.addOptionWithArg("max-failed-valid",
                        {"set number of fails on validation before exit."},
                        max_failed_vali);


  // --------------------------------------------------------
  // LineSearch extra options add by Salvatore Trani
  pmap.addMessage({"Training phase - specific options for Line Search:"});
  pmap.addOption("adaptive",
                 {"enable adaptive reduction factor",
                  "(based on last iteration metric gain)."});
  pmap.addOptionWithArg<std::string>("train-partial",
                                     {"set training file with partial scores",
                                      "(input for loading or output for saving)."});
  pmap.addOptionWithArg<std::string>("valid-partial",
                                     {"set validation file with partial scores",
                                      "(input for loading or output for saving)."});


  // --------------------------------------------------------
  // Optimization options add by Salvatore Trani
  pmap.addMessage({"Optimization phase - general options:"});
  pmap.addOptionWithArg<std::string>(
      "opt-algo",
      {"Optimization algorithm: [" +
          quickrank::optimization::post_learning::pruning::Cleaver::NAME_
           + "]."});

  std::string pruningMethods = "";
  for (auto
        i: quickrank::optimization::post_learning::pruning::Cleaver::pruningMethodNames) {
    pruningMethods += i + "|";
  }
  pruningMethods = pruningMethods.substr(0, pruningMethods.size() - 1);

  pmap.addOptionWithArg<std::string>(
      "opt-method",
      {"Optimization method: " +
          quickrank::optimization::post_learning::pruning::Cleaver::NAME_,
       "[" + pruningMethods + "]."});

  pmap.addOptionWithArg<std::string>(
      "opt-model",
      {"set output model file for optimization",
       "or input model file for testing."});

  pmap.addOptionWithArg<std::string>(
      "opt-algo-model",
      {"set output algorithm model file post optimization."});


  // --------------------------------------------------------
  pmap.addMessage({"Optimization phase - specific options for ensemble pruning:"});
  pmap.addOptionWithArg<double>("pruning-rate",
                                {"ensemble to prune (either as a ratio with",
                                 "respect to ensemble size or as an absolute",
                                 "number of estimators to prune)."});

  pmap.addOption("with-line-search",
                 {"ensemble pruning is made in conjunction",
                  "with line search [related parameters accepted]."});

  pmap.addOptionWithArg<std::string>("line-search-model",
                                     {"set line search XML file path for",
                                      "loading line search model (options",
                                      "and already trained weights)."});


  // --------------------------------------------------------
  pmap.addMessage({"Test phase - general options:"});
  pmap.addOptionWithArg("test-metric",
                        {"set test metric: ["
                             + quickrank::metric::ir::Dcg::NAME_
                             + "|"
                             + quickrank::metric::ir::Ndcg::NAME_ + "|"
                             + quickrank::metric::ir::Tndcg::NAME_ + "|"
                             + quickrank::metric::ir::Rmse::NAME_ + "|"
                             + quickrank::metric::ir::Map::NAME_ + "]."},
                        test_metric_string);

  pmap.addOptionWithArg("test-cutoff",
                        {"set test metric cutoff."},
                        test_cutoff);

  pmap.addOptionWithArg<std::string>("test", {"set testing file."});

  pmap.addOptionWithArg<std::string>("scores", {"set output scores file."});

  pmap.addOption("detailed",
                 {"enable detailed testing [applies only to ensemble models]."});


  // --------------------------------------------------------
  pmap.addMessage({"Code generation - general options:"});
  pmap.addOptionWithArg<std::string>("model-file",
                                     {"set XML model file path."});

  pmap.addOptionWithArg<std::string>("code-file", {"set C code file path."});

  pmap.addOptionWithArg("generator",
                        {"set C code generation strategy. Allowed options are:",
                         "-  \"condop\" (conditional operators),",
                         "-  \"oblivious\" (optimized code for oblivious trees),",
                         "-  \"vpred\" (intermediate code used by VPRED)."},
                        std::string("condop"));


  // --------------------------------------------------------
  pmap.addMessage({"Help options:"});
  pmap.addOption("help", "h", {"print help message."});


  bool parse_status = pmap.parse(argc, argv);
  if (!parse_status || pmap.isSet("help")) {
    std::cout << pmap.help();
    return EXIT_FAILURE;
  }

  return quickrank::driver::Driver::run(pmap);
}
