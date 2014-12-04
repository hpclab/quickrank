/*
 * QuickRank - A C++ suite of Learning to Rank algorithms
 * Webpage: http://quickrank.isti.cnr.it/
 * Contact: quickrank@isti.cnr.it
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
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
 *   - \b MatrixNet: I. Segalovich. Machine learning in search quality at yandex.
 *   Invited Talk, SIGIR, 2010.
 *
 * \subsection authors Authors and Contributors
 *
 * QuickRank has been developed by:
 *   - Claudio Lucchese (since Sept. 2014)
 *   - Franco Maria Nardini (since Sept. 2014)
 *   - Nicola Tonellotto (since Sept. 2014)
 *   - Gabriele Capannini (v0.0. June 2014 - Sept. 2014)
 *
 * \subsection download Get QuickRank
 * QuickRank is available here: \todo: put URL.
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
/// \todo TODO: (by cla) Find fastest sorting.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <time.h>
#include <iostream>
#include <iomanip>
#include <memory>

#include <boost/program_options.hpp>
#include <boost/algorithm/string/case_conv.hpp>

#include "metric/evaluator.h"
#include "learning/forests/mart.h"
#include "learning/forests/lambdamart.h"
#include "learning/forests/matrixnet.h"
#include "learning/custom/custom_ltr.h"
#include "metric/ir/tndcg.h"
#include "metric/ir/ndcg.h"
#include "metric/ir/map.h"
#include "io/xml.h"

namespace po = boost::program_options;


/// \todo TODO: To be moved elsewhere
std::shared_ptr<quickrank::metric::ir::Metric> metric_factory(
    std::string metric, unsigned int cutoff) {
  boost::to_upper(metric);
  if (metric == quickrank::metric::ir::Dcg::NAME_)
    return std::shared_ptr<quickrank::metric::ir::Metric>(
        new quickrank::metric::ir::Dcg(cutoff));
  else if (metric == quickrank::metric::ir::Ndcg::NAME_)
    return std::shared_ptr<quickrank::metric::ir::Metric>(
        new quickrank::metric::ir::Ndcg(cutoff));
  else if (metric == quickrank::metric::ir::Tndcg::NAME_)
    return std::shared_ptr<quickrank::metric::ir::Metric>(
        new quickrank::metric::ir::Tndcg(cutoff));
  else if (metric == quickrank::metric::ir::Map::NAME_)
    return std::shared_ptr<quickrank::metric::ir::Metric>(
        new quickrank::metric::ir::Map(cutoff));
  else
    return std::shared_ptr<quickrank::metric::ir::Metric>();
}

int main(int argc, char *argv[]) {
  std::cout << "# ## ================================== ## #" << std::endl
            << "# ##              QuickRank             ## #" << std::endl
            << "# ## ---------------------------------- ## #" << std::endl
            << "# ##     developed by the HPC. Lab.     ## #" << std::endl
            << "# ##      http://hpc.isti.cnr.it/       ## #" << std::endl
            << "# ##      quickrank@.isti.cnr.it        ## #" << std::endl
            << "# ## ================================== ## #" << std::endl;
  std::cout << std::fixed;

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
  std::string training_filename;
  std::string validation_filename;
  std::string test_filename;
  std::string features_filename;
  std::string model_filename;
  std::string xml_filename;
  std::string c_filename;
  std::string model_code_type;

  // Declare the supported options.
  po::options_description learning_desc("Training options");
  learning_desc.add_options()(
      "algo",
      po::value<std::string>(&algorithm_string)->default_value(
          algorithm_string),
      ("LtR algorithm [" + quickrank::learning::forests::Mart::NAME_ + "|"
          + quickrank::learning::forests::LambdaMart::NAME_ + "|"
          + quickrank::learning::forests::MatrixNet::NAME_ + "|"
          + quickrank::learning::CustomLTR::NAME_ + "]").c_str());
  learning_desc.add_options()(
      "train-metric",
      po::value<std::string>(&train_metric_string)->default_value(
          train_metric_string),
      ("set train metric [" + quickrank::metric::ir::Dcg::NAME_ + "|"
          + quickrank::metric::ir::Ndcg::NAME_ + "|"
          + quickrank::metric::ir::Tndcg::NAME_ + "|"
          + quickrank::metric::ir::Map::NAME_ + "]").c_str());
  learning_desc.add_options()(
      "train-cutoff",
      po::value<unsigned int>(&train_cutoff)->default_value(train_cutoff),
      "set train metric cutoff");
  learning_desc.add_options()(
      "test-metric",
      po::value<std::string>(&test_metric_string)->default_value(
          test_metric_string),
      ("set test metric [" + quickrank::metric::ir::Dcg::NAME_ + "|"
          + quickrank::metric::ir::Ndcg::NAME_ + "|"
          + quickrank::metric::ir::Tndcg::NAME_ + "|"
          + quickrank::metric::ir::Map::NAME_ + "]").c_str());
  learning_desc.add_options()(
      "test-cutoff",
      po::value<unsigned int>(&test_cutoff)->default_value(test_cutoff),
      "set test metric cutoff");
  learning_desc.add_options()(
      "partial",
      po::value<unsigned int>(&partial_save)->default_value(partial_save),
      "set partial file save frequency");
  learning_desc.add_options()(
      "train",
      po::value<std::string>(&training_filename)->default_value(
          training_filename),
      "set training file");
  learning_desc.add_options()(
      "valid",
      po::value<std::string>(&validation_filename)->default_value(
          validation_filename),
      "set validation file");
  learning_desc.add_options()(
      "test",
      po::value<std::string>(&test_filename)->default_value(test_filename),
      "set testing file");
  learning_desc.add_options()(
      "features",
      po::value<std::string>(&features_filename)->default_value(
          features_filename),
      "set features file");
  learning_desc.add_options()(
      "model",
      po::value<std::string>(&model_filename)->default_value(model_filename),
      "set output model file");

  po::options_description model_desc("Tree-based models options");
  model_desc.add_options()(
      "num-trees", po::value<unsigned int>(&ntrees)->default_value(ntrees),
      "set number of trees");
  model_desc.add_options()(
      "shrinkage", po::value<float>(&shrinkage)->default_value(shrinkage),
      "set shrinkage");
  model_desc.add_options()(
      "num-thresholds",
      po::value<unsigned int>(&nthresholds)->default_value(nthresholds),
      "set number of thresholds");
  model_desc.add_options()(
      "min-leaf-support",
      po::value<unsigned int>(&minleafsupport)->default_value(minleafsupport),
      "set minimum number of leaf support");
  model_desc.add_options()(
      "end-after-rounds",
      po::value<unsigned int>(&esr)->default_value(esr),
      "set num. rounds with no boost in validation before ending (if 0 disabled)");

  po::options_description lm_model_desc("Mart/LambdaMart options");
  lm_model_desc.add_options()(
      "num-leaves",
      po::value<unsigned int>(&ntreeleaves)->default_value(ntreeleaves),
      "set number of leaves");

  po::options_description mn_model_desc("MatrixNet options");
  mn_model_desc.add_options()(
      "tree-depth",
      po::value<unsigned int>(&treedepth)->default_value(treedepth),
      "set tree depth");

  po::options_description score_desc("Scoring options");
  score_desc.add_options()("dump-model",
                           po::value<std::string>(&xml_filename)->default_value(xml_filename),
                           "set XML model file path")(
      "dump-code", po::value<std::string>(&c_filename)->default_value(c_filename),
      "set C code file path")(
      "dump-type",
      po::value<std::string>(&model_code_type)->default_value("baseline"),
      "set C code generation strategy. Allowed options are: \"baseline\" and \"oblivious\".");

  po::options_description all_desc("Allowed options");
  all_desc.add(learning_desc).add(model_desc).add(lm_model_desc).add(
      mn_model_desc).add(score_desc);
  all_desc.add_options()("help,h", "produce help message");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, all_desc), vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cout << all_desc << "\n";
    return 1;
  }

  if (vm.count("algo")) {

    // Create model
    boost::to_upper(algorithm_string);
    std::shared_ptr<quickrank::learning::LTR_Algorithm> ranking_algorithm;
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
    else if (algorithm_string == quickrank::learning::forests::MatrixNet::NAME_)
      ranking_algorithm = std::shared_ptr<quickrank::learning::LTR_Algorithm>(
          new quickrank::learning::forests::MatrixNet(ntrees, shrinkage,
                                                      nthresholds, treedepth,
                                                      minleafsupport, esr));
    else if (algorithm_string == quickrank::learning::CustomLTR::NAME_)
      ranking_algorithm = std::shared_ptr<quickrank::learning::LTR_Algorithm>(
          new quickrank::learning::CustomLTR());
    else {
      std::cout << " !! Train Algorithm was not set properly" << std::endl;
      exit(EXIT_FAILURE);
    }

    // METRIC STUFF
    boost::to_upper(train_metric_string);
    std::shared_ptr<quickrank::metric::ir::Metric> training_metric =
        metric_factory(train_metric_string, train_cutoff);
    if (!training_metric) {
      std::cout << " !! Train Metric was not set properly" << std::endl;
      exit(EXIT_FAILURE);
    }

    boost::to_upper(test_metric_string);
    std::shared_ptr<quickrank::metric::ir::Metric> testing_metric =
        metric_factory(test_metric_string, test_cutoff);
    if (!testing_metric) {
      std::cout << " !! Test Metric was not set properly" << std::endl;
      exit(EXIT_FAILURE);
    }

    //show ranker parameters
    std::cout << "#" << std::endl << *ranking_algorithm;
    std::cout << "#" << std::endl << "# training scorer: " << *training_metric
              << std::endl << "# test scorer: " << *testing_metric << std::endl
              << "#" << std::endl;

    // FILE STUFF

    //set seed for rand()
    srand(time(NULL));

    quickrank::metric::Evaluator::evaluate(ranking_algorithm, training_metric,
                                           testing_metric, training_filename,
                                           validation_filename, test_filename,
                                           features_filename, model_filename,
                                           partial_save);

    return EXIT_SUCCESS;
  }

  // SCORING STUFF


  // if the dump files are set, it proceeds to dump the model by following a given strategy.
  if (xml_filename != "" && c_filename != "") {
    quickrank::io::Xml xml;
    if (model_code_type == "baseline") {
      std::cout << "applying baseline strategy for C code generation to: "
                << xml_filename << std::endl;
      xml.generate_c_code_baseline(xml_filename, c_filename);
      std::cout << "done.";
    } else
      std::cout << "applying oblivious strategy for C code generation to: "
                << xml_filename << std::endl;
    xml.generate_c_code_oblivious_trees(xml_filename, c_filename);
    std::cout << "done.";
  }

  return EXIT_SUCCESS;
}
