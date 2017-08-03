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
#include <iomanip>
#include <fstream>
#include <limits>
#include <numeric>
#include <io/generate_oblivious.h>
#include <learning/meta/meta_cleaver.h>

#include "driver/driver.h"
#include "io/svml.h"
#include "learning/ltr_algorithm_factory.h"
#include "optimization/optimization_factory.h"
#include "metric/metric_factory.h"
#include "utils/fileutils.h"

namespace quickrank {
namespace driver {

Driver::Driver() {
}

Driver::~Driver() {
}

int Driver::run(ParamsMap &pmap) {

  if (!pmap.isSet("train") && !pmap.isSet("train-partial") &&
      !pmap.isSet("test") && !pmap.isSet("model-file")) {
    std::cout << pmap.help();
    exit(EXIT_FAILURE);
  }

  if (pmap.isSet("train") || pmap.isSet("train-partial") ||
      pmap.isSet("test")) {

    std::shared_ptr<quickrank::learning::LTR_Algorithm> ranking_algorithm =
        quickrank::learning::ltr_algorithm_factory(pmap);
    if (!ranking_algorithm) {
      std::cerr << " !! LTR Algorithm was not set properly" << std::endl;
      exit(EXIT_FAILURE);
    }

    std::cout << std::endl << *ranking_algorithm << std::endl;

    // If there is the training dataset, it means we have to execute
    // the training phase and/or the optimization phase (at least one of them)
    if (pmap.isSet("train") || pmap.isSet("train-partial")) {

      std::shared_ptr<quickrank::optimization::Optimization> opt_algorithm;
      if ((pmap.isSet("opt-algo") || pmap.isSet("opt-model")) &&
          (ranking_algorithm->name() != learning::meta::MetaCleaver::NAME_
              || (!pmap.isSet("meta-algo") && !pmap.isSet("restart-train")) )) {

        opt_algorithm = quickrank::optimization::optimization_factory(pmap);
        if (!opt_algorithm) {
          std::cerr << " !! Optimization Algorithm was not set properly"
                    << std::endl;
          exit(EXIT_FAILURE);
        }

        std::cout << *opt_algorithm << std::endl;
      }

      std::string training_filename = pmap.get<std::string>("train");
      std::string validation_filename = pmap.get<std::string>("valid");
      std::string features_filename = pmap.get<std::string>("features");
      std::string model_filename_out = pmap.get<std::string>("model-out");
      std::string opt_model_filename = pmap.get<std::string>("opt-model");
      std::string opt_algo_model_filename =
          pmap.get<std::string>("opt-algo-model");
      size_t partial_save = pmap.get<std::size_t>("partial");
      std::string training_partial_filename;
      std::string validation_partial_filename;
      if (pmap.isSet("train-partial"))
        training_partial_filename = pmap.get<std::string>("train-partial");
      if (pmap.isSet("valid-partial"))
        validation_partial_filename = pmap.get<std::string>("valid-partial");

      std::shared_ptr<quickrank::data::Dataset> training_dataset;
      std::shared_ptr<quickrank::data::Dataset> validation_dataset;

      if (!training_filename.empty())
        training_dataset = load_dataset(training_filename, "training");

      if (!validation_filename.empty())
        validation_dataset = load_dataset(validation_filename, "validation");

      if (!features_filename.empty()) {
        // TODO: filter features while loading dataset
      }

      std::shared_ptr<quickrank::metric::ir::Metric> training_metric =
          quickrank::metric::ir::ir_metric_factory(
              pmap.get<std::string>("train-metric"),
              pmap.get<size_t>("train-cutoff"));
      if (!training_metric) {
        std::cerr << " !! Train Metric was not set properly" << std::endl;
        exit(EXIT_FAILURE);
      }

      if (opt_algorithm && opt_algorithm->is_pre_learning()) {
        // We have to run the optimization process pre-training
        optimization_phase(opt_algorithm,
                           ranking_algorithm,
                           training_metric,
                           training_dataset,
                           validation_dataset,
                           training_partial_filename,
                           validation_partial_filename,
                           opt_model_filename,
                           opt_algo_model_filename,
                           partial_save);
      }

      // If the training algorithm has been created from scratch (not loaded
      // from file), we have to run the training phase
      if (pmap.isSet("train") && !pmap.isSet("skip-train") && (
          !pmap.isSet("model-in") || pmap.isSet("restart-train")) ) {

        //show ranker parameters
        std::cout << "#" << std::endl << *ranking_algorithm;
        std::cout << "#" << std::endl << "# training scorer: "
                  << *training_metric
                  << std::endl;

        training_phase(ranking_algorithm,
                       training_metric,
                       training_dataset,
                       validation_dataset,
                       model_filename_out,
                       partial_save);
      }

      if (opt_algorithm && !opt_algorithm->is_pre_learning()) {
        // We have to run the optimization process post-training
        optimization_phase(opt_algorithm,
                           ranking_algorithm,
                           training_metric,
                           training_dataset,
                           validation_dataset,
                           training_partial_filename,
                           validation_partial_filename,
                           opt_model_filename,
                           opt_algo_model_filename,
                           partial_save);
      }
    }

    if (pmap.isSet("test")) {
      std::string test_filename = pmap.get<std::string>("test");
      std::string scores_filename = pmap.get<std::string>("scores");
      bool detailed_testing = pmap.isSet("detailed");

      std::shared_ptr<quickrank::data::Dataset> test_dataset;
      if (!test_filename.empty())
        test_dataset = load_dataset(test_filename, "testing");

      std::shared_ptr<quickrank::metric::ir::Metric> testing_metric =
          quickrank::metric::ir::ir_metric_factory(
              pmap.get<std::string>("test-metric"),
              pmap.get<size_t>("test-cutoff"));
      if (!testing_metric) {
        std::cerr << " !! Test Metric was not set properly" << std::endl;
        exit(EXIT_FAILURE);
      }

      std::cout << "# test scorer: " << *testing_metric << std::endl << "#" <<
                std::endl;
      testing_phase(ranking_algorithm,
                    testing_metric,
                    test_dataset,
                    scores_filename,
                    detailed_testing);
    }
  }

  // Code Generation
  // if the dump files are set, it proceeds to dump the model by following a given strategy.
  if (pmap.count("model-file") && pmap.count("code-file")) {
    std::string xml_filename = pmap.get<std::string>("model-file");
    std::string c_filename = pmap.get<std::string>("code-file");
    std::string generator_type = pmap.get<std::string>("generator");

    if (generator_type == "condop") {
      quickrank::io::GenOpCond conditional_operator_generator;
      std::cout
          << "applying conditional operators strategy for C code generation to: "
          << xml_filename << std::endl;
      conditional_operator_generator.generate_conditional_operators_code(
          xml_filename,
          c_filename);
    } else if (generator_type == "oblivious") {
      quickrank::io::GenOblivious oblivious_generator;
      std::cout << "applying oblivious strategy for C code generation to: "
                << xml_filename << std::endl;
      oblivious_generator.generate_oblivious_code(xml_filename, c_filename);
    } else if (generator_type == "vpred") {
      quickrank::io::GenVpred vpred_generator;
      std::cout << "generating VPred input file from: " << xml_filename
                << std::endl;
      vpred_generator.generate_vpred_input(xml_filename, c_filename);
    }
  }

  return EXIT_SUCCESS;
}

void Driver::training_phase(
    std::shared_ptr<learning::LTR_Algorithm> algo,
    std::shared_ptr<quickrank::metric::ir::Metric> train_metric,
    std::shared_ptr<quickrank::data::Dataset> training_dataset,
    std::shared_ptr<quickrank::data::Dataset> validation_dataset,
    const std::string output_filename,
    const size_t npartialsave) {

  // run the learning process
  algo->learn(training_dataset, validation_dataset, train_metric, npartialsave,
              output_filename);

  if (!output_filename.empty()) {
    std::cout << std::endl;
    std::cout << "# Writing model to file: " << output_filename
              << std::endl << std::endl;
    algo->save(output_filename);
  }
}

void Driver::optimization_phase(
    std::shared_ptr<quickrank::optimization::Optimization> opt_algorithm,
    std::shared_ptr<learning::LTR_Algorithm> ranking_algo,
    std::shared_ptr<quickrank::metric::ir::Metric> train_metric,
    std::shared_ptr<quickrank::data::Dataset> training_dataset,
    std::shared_ptr<quickrank::data::Dataset> validation_dataset,
    std::string training_partial_filename,
    std::string validation_partial_filename,
    const std::string output_filename,
    const std::string opt_algo_model_filename,
    const size_t npartialsave) {

  std::shared_ptr<quickrank::data::Dataset> training_partial_dataset;
  std::shared_ptr<quickrank::data::Dataset> validation_partial_dataset;

  // Variable meaning the algo needs partial scores
  bool need_ps = opt_algorithm->need_partial_score_dataset();

  if (opt_algorithm && need_ps) {

    if (!training_partial_filename.empty() &&
        file_exist(training_partial_filename))
      training_partial_dataset = load_dataset(training_partial_filename,
                                              "training (partial)");

    if (!validation_partial_filename.empty() &&
        file_exist(validation_partial_filename))
      validation_partial_dataset = load_dataset(validation_partial_filename,
                                                "validation (partial)");

    quickrank::io::Svml svml;
    if (!training_partial_dataset && training_dataset) {

      training_partial_dataset = Driver::extract_partial_scores(
          ranking_algo,
          training_dataset,
          true);

      if (!training_partial_filename.empty())
        svml.write(training_partial_dataset, training_partial_filename);
    }

    if (!validation_partial_dataset && validation_dataset) {

      validation_partial_dataset = Driver::extract_partial_scores(
          ranking_algo,
          validation_dataset,
          true);

      if (!validation_partial_filename.empty())
        svml.write(validation_partial_dataset, validation_partial_filename);
    }
  }

  // run the optimization process
  opt_algorithm->optimize(
      ranking_algo,
      need_ps ? training_partial_dataset : training_dataset,
      need_ps ? validation_partial_dataset : validation_dataset,
      train_metric,
      npartialsave,
      output_filename);

  if (!output_filename.empty()) {
    std::cout << std::endl;
    std::cout << "# Writing optimization model to file: "
              << output_filename << std::endl;
    opt_algorithm->save(output_filename);
  }

  if (!opt_algo_model_filename.empty()) {
    std::cout << std::endl;
    std::cout << "# Writing optimized LTR algo model to file: "
              << opt_algo_model_filename << std::endl << std::endl;
    ranking_algo->save(opt_algo_model_filename);
  }
}

void Driver::testing_phase(
    std::shared_ptr<learning::LTR_Algorithm> algo,
    std::shared_ptr<quickrank::metric::ir::Metric> test_metric,
    std::shared_ptr<quickrank::data::Dataset> test_dataset,
    const std::string scores_filename,
    const bool detailed_testing) {

  if (test_metric and test_dataset) {

    std::vector<Score> scores(test_dataset->num_instances(), 0.0);
    if (detailed_testing) {
      std::shared_ptr<data::Dataset> datasetPartScores =
          Driver::extract_partial_scores(algo, test_dataset);

      Feature *features = datasetPartScores->at(0, 0);
#pragma omp parallel for
      for (unsigned int s = 0; s < datasetPartScores->num_instances(); ++s) {
        size_t offset_feature = s * datasetPartScores->num_features();
        // compute partialScore * weight for all the trees
        for (unsigned int f = 0; f < datasetPartScores->num_features(); ++f) {
          scores[s] += features[offset_feature + f];
        }
      }

      quickrank::MetricScore test_score = test_metric->evaluate_dataset(
          test_dataset, &scores[0]);

      std::cout << *test_metric << " on test data = " << std::setprecision(4)
                << test_score << std::endl << std::endl;

      quickrank::io::Svml svml;
      svml.write(datasetPartScores, scores_filename);

      std::cout << "# Partial Scores written to file: " << scores_filename
                << std::endl;

    } else {
      algo->score_dataset(test_dataset, &scores[0]);
      quickrank::MetricScore test_score = test_metric->evaluate_dataset(
          test_dataset, &scores[0]);

      std::cout << std::endl;
      std::cout << *test_metric << " on test data = " << std::setprecision(4)
                << test_score << std::endl << std::endl;

      if (!scores_filename.empty()) {
        std::ofstream os;
        os << std::setprecision(std::numeric_limits<Score>::max_digits10);
        os.open(scores_filename, std::fstream::out);
        for (size_t i = 0; i < test_dataset->num_instances(); ++i)
          os << scores[i] << std::endl;
        os.close();
        std::cout << "# Scores written to file: " << scores_filename
                  << std::endl;
      }
    }
  }

  algo->print_additional_stats();
}

std::shared_ptr<quickrank::data::Dataset> Driver::load_dataset(
    const std::string dataset_filename,
    const std::string dataset_label) {

  // create reader: assume svml as ltr format
  quickrank::io::Svml reader;

  std::shared_ptr<quickrank::data::Dataset> dataset = nullptr;
  if (!dataset_filename.empty()) {
    std::cout << "# Reading " + dataset_label + " dataset: " <<
              dataset_filename << std::endl;
    dataset = reader.read_horizontal(dataset_filename);
    std::cout << reader << *dataset << std::endl;
  }

  if (!dataset) {
    std::cerr << "!!! Error while loading " + dataset_label + " dataset" <<
              std::endl;
    exit(EXIT_FAILURE);
  }

  return dataset;
}

std::shared_ptr<data::Dataset> Driver::extract_partial_scores(
    std::shared_ptr<learning::LTR_Algorithm> algo,
    std::shared_ptr<data::Dataset> dataset,
    bool ignore_weights) {

  data::Dataset *datasetPartScores = nullptr;
  for (size_t q = 0; q < dataset->num_queries(); q++) {
    auto results = dataset->getQueryResults(q);
    // score_query_results(r, scores, 1, test_dataset->num_features());
    const Feature *features = results->features();
    const Label *labels = results->labels();
    for (size_t i = 0; i < results->num_results(); i++) {
      auto detailed_scores = algo->partial_scores_document(features,
                                                           ignore_weights);

      if (!detailed_scores) {
        std::cerr << "# ## ERROR!! Only Ensemble methods support the "
                  << "export of detailed score tree by tree" << std::endl;
        exit(EXIT_FAILURE);
      }

      // Initilizing on the first instance of the dataset
      if (datasetPartScores == nullptr)
        datasetPartScores = new data::Dataset(dataset->num_instances(),
                                              detailed_scores->size());
      // It performs a copy for casting Score to Feature (double to float)
      std::vector<Feature> featuresScore(detailed_scores->begin(),
                                         detailed_scores->end());
      datasetPartScores->addInstance(q, labels[i], featuresScore);

      features += dataset->num_features();
    }
  }

  return std::shared_ptr<data::Dataset>(datasetPartScores);
}

}  // namespace driver
}  // namespace quickrank
