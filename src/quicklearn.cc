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
 * \mainpage QuickRank: Efficient Learning-to-Rank Toolkit
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
 *   - \b Oblivious GBRT / LambdaMart: Inspired to I. Segalovich. Machine learning in search quality at yandex.
 *   Invited Talk, SIGIR, 2010.
 *   - \b CoordinateAscent: Metzler, D., Croft, W.B.: Linear feature-based models for information retrieval.
 *   Information Retrieval 10(3), 257–274 (2007)
 *
 * \subsection authors Authors and Contributors
 *
 * QuickRank has been developed by:
 *   - Claudio Lucchese (since Sept. 2014)
 *   - Franco Maria Nardini (since Sept. 2014)
 *   - Nicola Tonellotto (since Sept. 2014)
 *   - Gabriele Capannini (v0.0. June 2014 - Sept. 2014)
 *   - Andrea Battistini and Chiara Pierucci (Implementation of CoordinateAscent - March 2015)
 *
 *
 * \subsection download Get QuickRank
 * QuickRank is available here: <a href="http://quickrank.isti.cnr.it">http://quickrank.isti.cnr.it</a>.
 *
 * \section Usage
 *
 * \subsection cmd Command line options
 *
 * \todo command line description
 *
 * \subsection compile Compilation
 *
 *
 * \section log ChangeLog
 *
 * - xx/xx/2014: Version 1.1 released
 *
 *
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

#include <boost/program_options.hpp>
#include <boost/algorithm/string/case_conv.hpp>

#include "metric/evaluator.h"
#include "learning/forests/mart.h"
#include "learning/forests/lambdamart.h"
#include "learning/forests/obliviousmart.h"
#include "learning/forests/obliviouslambdamart.h"
#include "learning/linear/coordinate_ascent.h"
#include "learning/linear/line_search.h"
#include "pruning/ensemble_pruning.h"
#include "learning/custom/custom_ltr.h"
#include "metric/metricfactory.h"
#include "io/xml.h"
#include "scoring/opt/converter.h"

namespace po = boost::program_options;



void print_logo() {
  if ( isatty(fileno(stdout)) ) {
    std::string color_reset = "\033[0m";
    std::string color_logo = "\033[1m\033[32m";
    std::cout << color_logo << std::endl
              << "      _____  _____" << std::endl
              << "     /    / /____/"  << std::endl
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
  unsigned int ntrees = 1000;
  float shrinkage = 0.10f;
  unsigned int nthresholds = 0;
  unsigned int minleafsupport = 1;
  unsigned int esr = 100;
  unsigned int ntreeleaves = 10;
  unsigned int treedepth = 3;
  std::string train_metric_string = "NDCG";
  unsigned int train_cutoff = 10;
  std::string test_metric_string = "NDCG";
  unsigned int test_cutoff = 10;
  unsigned int partial_save = 100;
  bool verbose_testing = false;
  bool ensemble_pruning_with_line_search = false;
  std::string training_filename;
  std::string validation_filename;
  std::string test_filename;
  std::string features_filename;
  std::string model_filename;
  std::string scores_filename;
  std::string xml_filename;
  std::string c_filename;
  std::string model_code_type;

  // ------------------------------------------
  // Coordinate ascent added by Chiara Pierucci
  unsigned int num_points = 21;
  unsigned int max_iterations = 100;
  float window_size = 10.0;
  float reduction_factor = 0.95;
  unsigned int max_failed_vali = 20;

  // ------------------------------------------
  // Ensemble Pruning added by Salvatore Trani
  double epruning_rate = 0.5;
  std::string epruning_method = "RANDOM";

  // data structures
  std::shared_ptr<quickrank::learning::LTR_Algorithm> ranking_algorithm;

  // Declare the supported options.
  po::options_description learning_options("Training options");
  learning_options.add_options()(
      "algo",
      po::value<std::string>(&algorithm_string)->default_value(
          algorithm_string),
      ("LtR algorithm ["
          + quickrank::learning::forests::Mart::NAME_ + "|"
          + quickrank::learning::forests::LambdaMart::NAME_ + "|"
          + quickrank::learning::forests::ObliviousMart::NAME_ + "|"
          + quickrank::learning::forests::ObliviousLambdaMart::NAME_ + "|"
          + quickrank::learning::linear::CoordinateAscent::NAME_ + "|"
          + quickrank::learning::linear::LineSearch::NAME_ + "|"
          + quickrank::pruning::EnsemblePruning::NAME_ + "|"
          + quickrank::learning::CustomLTR::NAME_ + "]")
          .c_str());
  learning_options.add_options()(
      "train-metric",
      po::value<std::string>(&train_metric_string)->default_value(
          train_metric_string),
      ("set train metric [" + quickrank::metric::ir::Dcg::NAME_ + "|"
          + quickrank::metric::ir::Ndcg::NAME_ + "|"
          + quickrank::metric::ir::Tndcg::NAME_ + "|"
          + quickrank::metric::ir::Map::NAME_ + "]").c_str());
  learning_options.add_options()(
      "train-cutoff",
      po::value<unsigned int>(&train_cutoff)->default_value(train_cutoff),
      "set train metric cutoff");
  learning_options.add_options()(
      "partial",
      po::value<unsigned int>(&partial_save)->default_value(partial_save),
      "set partial file save frequency");
  learning_options.add_options()(
      "train",
      po::value<std::string>(&training_filename)->default_value(
          training_filename),
      "set training file");
  learning_options.add_options()(
      "valid",
      po::value<std::string>(&validation_filename)->default_value(
          validation_filename),
      "set validation file");
  learning_options.add_options()(
      "features",
      po::value<std::string>(&features_filename)->default_value(
          features_filename),
      "set features file");
  learning_options.add_options()(
      "model",
      po::value<std::string>(&model_filename)->default_value(model_filename),
      "set output model file for training or input model file for testing");

  po::options_description tree_model_options(
      "Training options for tree-based models");
  tree_model_options.add_options()(
      "num-trees", po::value<unsigned int>(&ntrees)->default_value(ntrees),
      "set number of trees");
  tree_model_options.add_options()(
      "shrinkage", po::value<float>(&shrinkage)->default_value(shrinkage),
      "set shrinkage");
  tree_model_options.add_options()(
      "num-thresholds",
      po::value<unsigned int>(&nthresholds)->default_value(nthresholds),
      "set number of thresholds");
  tree_model_options.add_options()(
      "min-leaf-support",
      po::value<unsigned int>(&minleafsupport)->default_value(minleafsupport),
      "set minimum number of leaf support");
  tree_model_options.add_options()(
      "end-after-rounds",
      po::value<unsigned int>(&esr)->default_value(esr),
      "set num. rounds with no boost in validation before ending (if 0 disabled)");
  tree_model_options.add_options()(
      "num-leaves",
      po::value<unsigned int>(&ntreeleaves)->default_value(ntreeleaves),
      "set number of leaves [applies only to Mart/LambdaMart]");
  tree_model_options.add_options()(
      "tree-depth",
      po::value<unsigned int>(&treedepth)->default_value(treedepth),
      "set tree depth [applies only to Oblivious Mart/LambdaMart]");

  po::options_description testing_options("Testing options");
  testing_options.add_options()(
      "test-metric",
      po::value<std::string>(&test_metric_string)->default_value(
          test_metric_string),
      ("set test metric [" + quickrank::metric::ir::Dcg::NAME_ + "|"
          + quickrank::metric::ir::Ndcg::NAME_ + "|"
          + quickrank::metric::ir::Tndcg::NAME_ + "|"
          + quickrank::metric::ir::Map::NAME_ + "]").c_str());
  testing_options.add_options()(
      "test-cutoff",
      po::value<unsigned int>(&test_cutoff)->default_value(test_cutoff),
      "set test metric cutoff");
  testing_options.add_options()(
      "test",
      po::value<std::string>(&test_filename)->default_value(test_filename),
      "set testing file");
  testing_options.add_options()(
      "scores",
      po::value<std::string>(&scores_filename)->default_value(scores_filename),
      "set output scores file");
  testing_options.add_options()(
      "verbose",
      po::bool_switch(&verbose_testing),
      "set verbose testing [applies only to ensemble models]");

  po::options_description fast_scoring_options("Fast Scoring options");
  fast_scoring_options.add_options()(
      "dump-model",
      po::value<std::string>(&xml_filename)->default_value(xml_filename),
      "set XML model file path")(
      "dump-code",
      po::value<std::string>(&c_filename)->default_value(c_filename),
      "set C code file path")(
      "dump-type",
      po::value<std::string>(&model_code_type)->default_value("baseline"),
      "set C code generation strategy. Allowed options are: \"baseline\", \"oblivious\". \"vpred\".");

  // CoordinateAscent options add by Chiara Pierucci
  po::options_description coordasc_options(
      "Training options for Coordinate Ascent and Line Search");
  coordasc_options.add_options()(
      "num-samples",
      po::value<unsigned int>(&num_points)->default_value(num_points),
      "set number of samples in search window");
  coordasc_options.add_options()(
      "window-size", po::value<float>(&window_size)->default_value(window_size),
      "set search window size");
  coordasc_options.add_options()(
      "reduction-factor",
      po::value<float>(&reduction_factor)->default_value(reduction_factor),
      "set window reduction factor");
  coordasc_options.add_options()(
      "max-iterations",
      po::value<unsigned int>(&max_iterations)->default_value(max_iterations),
      "set number of max iterations");
  coordasc_options.add_options()(
      "max-failed-valid",
      po::value<unsigned int>(&max_failed_vali)->default_value(max_failed_vali),
      "set number of fails on validation before exit");

  // Ensemble Pruning options add by Salvatore Trani
  po::options_description epruning_options(
      "Training options for Ensemble Pruning");
  epruning_options.add_options()(
      "pruning-rate",
      po::value<double>(&epruning_rate)->default_value(epruning_rate),
      "ensemble to prune (either as a ratio with respect to ensemble size or "
          "as an absolute number of estimators to prune)");
  epruning_options.add_options()(
      "pruning-method",
      po::value<std::string>(&epruning_method)->default_value(epruning_method),
      "method for pruning the ensemble");
  epruning_options.add_options()(
      "with-line-search",
      po::bool_switch(&ensemble_pruning_with_line_search),
      "ensemble pruning is made in conjunction with line search "
          "[related parameters accepted]");

  po::options_description all_desc("Allowed options");
  all_desc.add(learning_options)
      .add(tree_model_options)
      .add(coordasc_options)
      .add(epruning_options)
      .add(testing_options)
      .add(fast_scoring_options);
  all_desc.add_options()("help,h", "produce help message");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, all_desc), vm);
  po::notify(vm);

  // Show Help
  if (vm.count("help")) {
    std::cout << all_desc << "\n";
    return 1;
  }

  // Run Training
  if (!training_filename.empty()) {

    // Create model
    boost::to_upper(algorithm_string);
    if (algorithm_string == quickrank::learning::forests::LambdaMart::NAME_)
      ranking_algorithm = std::shared_ptr<quickrank::learning::LTR_Algorithm>(
          new quickrank::learning::forests::LambdaMart(ntrees, shrinkage,
                                                       nthresholds, ntreeleaves,
                                                       minleafsupport, esr));
    else if (algorithm_string == quickrank::learning::forests::Mart::NAME_)
      ranking_algorithm = std::shared_ptr<quickrank::learning::LTR_Algorithm>(
          new quickrank::learning::forests::Mart(ntrees, shrinkage, nthresholds,
                                                 ntreeleaves, minleafsupport,
                                                 esr));
    else if (algorithm_string == quickrank::learning::forests::ObliviousMart::NAME_)
      ranking_algorithm = std::shared_ptr<quickrank::learning::LTR_Algorithm>(
          new quickrank::learning::forests::ObliviousMart(ntrees, shrinkage,
                                                      nthresholds, treedepth,
                                                      minleafsupport, esr));
    else if (algorithm_string == quickrank::learning::forests::ObliviousLambdaMart::NAME_)
      ranking_algorithm = std::shared_ptr<quickrank::learning::LTR_Algorithm>(
          new quickrank::learning::forests::ObliviousLambdaMart(ntrees, shrinkage,
                                                      nthresholds, treedepth,
                                                      minleafsupport, esr));
    else if (algorithm_string == quickrank::learning::linear::CoordinateAscent::NAME_)
      ranking_algorithm = std::shared_ptr<quickrank::learning::LTR_Algorithm>(
          new quickrank::learning::linear::CoordinateAscent(num_points,
                                                            window_size,
                                                            reduction_factor,
                                                            max_iterations,
                                                            max_failed_vali));
    else if (algorithm_string == quickrank::learning::linear::LineSearch::NAME_)
      ranking_algorithm = std::shared_ptr<quickrank::learning::LTR_Algorithm>(
          new quickrank::learning::linear::LineSearch(num_points,
                                                      window_size,
                                                      reduction_factor,
                                                      max_iterations,
                                                      max_failed_vali));
    else if (algorithm_string == quickrank::pruning::EnsemblePruning::NAME_) {
      if (!ensemble_pruning_with_line_search)
        ranking_algorithm = std::shared_ptr<quickrank::learning::LTR_Algorithm>(
          new quickrank::pruning::EnsemblePruning(epruning_method,
                                                  epruning_rate));
      else {
        std::shared_ptr<quickrank::learning::linear::LineSearch> lineSearch =
            std::shared_ptr<quickrank::learning::linear::LineSearch>(
                new quickrank::learning::linear::LineSearch(
                    num_points, window_size, reduction_factor, max_iterations,
                    max_failed_vali));
        ranking_algorithm = std::shared_ptr<quickrank::learning::LTR_Algorithm>(
            new quickrank::pruning::EnsemblePruning(epruning_method,
                                                    epruning_rate, lineSearch));
      }
    } else if (algorithm_string == quickrank::learning::CustomLTR::NAME_)
      ranking_algorithm = std::shared_ptr<quickrank::learning::LTR_Algorithm>(
          new quickrank::learning::CustomLTR());
    else {
      std::cout << " !! Train Algorithm was not set properly" << std::endl;
      exit(EXIT_FAILURE);
    }

    // METRIC STUFF
    std::shared_ptr<quickrank::metric::ir::Metric> training_metric =
        quickrank::metric::ir::ir_metric_factory(train_metric_string, train_cutoff);
    if (!training_metric) {
      std::cout << " !! Train Metric was not set properly" << std::endl;
      exit(EXIT_FAILURE);
    }

    //show ranker parameters
    std::cout << "#" << std::endl << *ranking_algorithm;
    std::cout << "#" << std::endl << "# training scorer: " << *training_metric
              << std::endl;

    quickrank::metric::Evaluator::training_phase(ranking_algorithm,
                                                 training_metric,
                                                 training_filename,
                                                 validation_filename,
                                                 features_filename,
                                                 model_filename, partial_save);
  }

  if (!test_filename.empty()) {
    if (!ranking_algorithm) {
      std::cout << "# Loading model from file " << model_filename << std::endl;
      ranking_algorithm =
          quickrank::learning::LTR_Algorithm::load_model_from_file(
              model_filename);
      std::cout << "#" << std::endl << *ranking_algorithm;
      if (!ranking_algorithm) {
        std::cout << " !! Unable to load model from file." << std::endl;
        exit(EXIT_FAILURE);
      }
    }

    std::shared_ptr<quickrank::metric::ir::Metric> testing_metric =
        quickrank::metric::ir::ir_metric_factory(test_metric_string, test_cutoff);
    if (!testing_metric) {
      std::cout << " !! Test Metric was not set properly" << std::endl;
      exit(EXIT_FAILURE);
    }

    std::cout << "# test scorer: " << *testing_metric << std::endl << "#"
              << std::endl;
    quickrank::metric::Evaluator::testing_phase(ranking_algorithm,
                                                testing_metric, test_filename,
                                                scores_filename,
                                                verbose_testing);
  }

  // Fast Scoring

  // if the dump files are set, it proceeds to dump the model by following a given strategy.
  if (xml_filename != "" && c_filename != "") {
    quickrank::io::Xml xml;
    if (model_code_type == "baseline") {
      std::cout << "applying baseline strategy (conditional operators) for C code generation to: "
                << xml_filename << std::endl;
      xml.generate_c_code_baseline(xml_filename, c_filename);
    } else if (model_code_type == "oblivious") {
      std::cout << "applying oblivious strategy for C code generation to: "
                << xml_filename << std::endl;
      xml.generate_c_code_oblivious_trees(xml_filename, c_filename);
    } else if (model_code_type == "vpred") {
      std::cout << "generating VPred input file to: " << "stdout."
                << std::endl;
      quickrank::scoring::generate_opt_trees_input(xml_filename, c_filename);
    }
  }

  return EXIT_SUCCESS;
}
