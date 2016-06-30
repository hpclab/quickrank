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

#include "driver/driver.h"
#include "io/svml.h"

#include "learning/ltr_algorithm_factory.h"
#include "metric/metric_factory.h"

namespace quickrank {
namespace driver {

Driver::Driver() {
}

  Driver::~Driver() {
}

int Driver::run(ParamsMap& pmap) {

  std::shared_ptr<quickrank::learning::LTR_Algorithm> ranking_algorithm =
      quickrank::learning::ltr_algorithm_factory(pmap);
  if (!ranking_algorithm) {
    std::cerr << " !! LTR Algorithm was not set properly" << std::endl;
    exit(EXIT_FAILURE);
  }

  if (pmap.count("train")) {
    std::string training_filename = pmap.get<std::string>("train");
    std::string validation_filename = pmap.get<std::string>("valid");
    std::string features_filename = pmap.get<std::string>("features");
    std::string model_filename = pmap.get<std::string>("model");
    size_t partial_save = pmap.get<std::size_t>("partial");

    std::shared_ptr<quickrank::metric::ir::Metric> training_metric =
        quickrank::metric::ir::ir_metric_factory(
            pmap.get<std::string>("train-metric"),
            pmap.get<size_t>("train-cutoff"));
    if (!training_metric) {
      std::cerr << " !! Train Metric was not set properly" << std::endl;
      exit(EXIT_FAILURE);
    }

    //show ranker parameters
    std::cout << "#" << std::endl << *ranking_algorithm;
    std::cout << "#" << std::endl << "# training scorer: " << *training_metric
    << std::endl;

    training_phase(ranking_algorithm,
                   training_metric,
                   training_filename,
                   validation_filename,
                   features_filename,
                   model_filename,
                   partial_save);
  }

  if (pmap.isSet("test")) {
    std::string test_filename = pmap.get<std::string>("test");
    std::string scores_filename = pmap.get<std::string>("scores");
    bool detailed_testing = pmap.isSet("detailed");

    std::shared_ptr<quickrank::metric::ir::Metric> testing_metric =
        quickrank::metric::ir::ir_metric_factory(
          pmap.get<std::string>("test-metric"),
          pmap.get<size_t>("test-cutoff"));
    if (!testing_metric) {
      std::cerr << " !! Train Metric was not set properly" << std::endl;
      exit(EXIT_FAILURE);
    }

    std::cout << "# test scorer: " << *testing_metric << std::endl << "#"
    << std::endl;
    testing_phase(ranking_algorithm,
                  testing_metric,
                  test_filename,
                  scores_filename,
                  detailed_testing);
  }

  // Fast Scoring

  // if the dump files are set, it proceeds to dump the model by following a given strategy.
  if (pmap.count("dump-model") && pmap.count("dump-code")) {
    std::string xml_filename = pmap.get<std::string>("dump-model");
    std::string c_filename = pmap.get<std::string>("dump-code");
    std::string model_code_type = pmap.get<std::string>("dump-type");

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
      std::cout << "generating VPred input file from: " << xml_filename
        << std::endl;
      quickrank::io::generate_vpred_input(xml_filename, c_filename);
    }
  }

  return EXIT_SUCCESS;
}

void Driver::training_phase(
    std::shared_ptr<learning::LTR_Algorithm> algo,
    std::shared_ptr<quickrank::metric::ir::Metric> train_metric,
    const std::string training_filename,
    const std::string validation_filename,
    const std::string feature_filename,
    const std::string output_filename,
    const size_t npartialsave) {

  // create reader: assum svml as ltr format
  quickrank::io::Svml reader;

  std::shared_ptr<quickrank::data::Dataset> training_dataset;
  std::shared_ptr<quickrank::data::Dataset> validation_dataset;

  if (!training_filename.empty()) {
    std::cout << "# Reading training dataset: " << training_filename
              << std::endl;
    training_dataset = reader.read_horizontal(training_filename);
    std::cout << reader << *training_dataset;
  } else {
    std::cerr << "!!! Error while loading training dataset" << std::endl;
    exit(EXIT_FAILURE);
  }

  if (!validation_filename.empty()) {
    std::cout << "# Reading validation dataset: " << validation_filename
              << std::endl;
    validation_dataset = reader.read_horizontal(validation_filename);
    std::cout << reader << *validation_dataset;
  }

  if (!feature_filename.empty()) {
    /// \todo TODO: filter features while loading dataset
  }

  // run the learning process
  algo->learn(training_dataset, validation_dataset, train_metric, npartialsave,
              output_filename);

  if (!output_filename.empty()) {
    std::cout << std::endl;
    std::cout << "# Writing model to file: " << output_filename << std::endl;
    algo->save(output_filename);
  }
}

void Driver::testing_phase(
    std::shared_ptr<learning::LTR_Algorithm> algo,
    std::shared_ptr<quickrank::metric::ir::Metric>
    test_metric,
    const std::string test_filename,
    const std::string scores_filename,
    const bool detailed_testing) {

  if (test_metric and !test_filename.empty()) {

    // create reader: assume svml as ltr format
    quickrank::io::Svml svml;

    std::cout << "# Reading test dataset: " << test_filename << std::endl;

    std::shared_ptr<data::Dataset> test_dataset =
        svml.read_horizontal(test_filename);
    std::cout << svml << *test_dataset << std::endl;

    std::vector<Score> scores(test_dataset->num_instances());
    if (detailed_testing) {
      size_t idx_query_scores = 0;
      std::shared_ptr<data::Dataset> datasetPartScores = nullptr;

      for (size_t q = 0; q < test_dataset->num_queries(); q++) {
        std::shared_ptr<data::QueryResults> results =
            test_dataset->getQueryResults(q);
        // score_query_results(r, scores, 1, test_dataset->num_features());
        const Feature* features = results->features();
        const Label* labels = results->labels();
        for (size_t i = 0; i < results->num_results(); i++) {
          std::shared_ptr<std::vector<Score>> detailed_scores =
              algo->detailed_scores_document(features);

          if (detailed_scores == nullptr) {
            std::cerr << "# ## ERROR!! Only Ensemble methods support the " <<
                "export of detailed score tree by tree" << std::endl;
            return;
          }

          // Initilized on iterating the first instance in the dataset
          if (datasetPartScores == nullptr)
            datasetPartScores = std::shared_ptr<data::Dataset>(
                new data::Dataset(test_dataset->num_instances(),
                                  detailed_scores->size()));
          // It performs a copy for casting Score to Feature (double to float)
          std::vector<Feature> featuresScore(detailed_scores->begin(),
                                             detailed_scores->end());
          datasetPartScores->addInstance(q, labels[i], featuresScore);

          scores[idx_query_scores + i] = std::accumulate(
              detailed_scores->begin(), detailed_scores->end(), 0.0);
          features += test_dataset->num_features();
        }
        idx_query_scores += results->num_results();
      }

      quickrank::MetricScore test_score = test_metric->evaluate_dataset(
              test_dataset, &scores[0]);

      std::cout << *test_metric << " on test data = " << std::setprecision(4)
        << test_score << std::endl << std::endl;


      svml.write(datasetPartScores, scores_filename);

    } else {
      algo->score_dataset(test_dataset, &scores[0]);
      quickrank::MetricScore test_score = test_metric->evaluate_dataset(
              test_dataset, &scores[0]);

      std::cout << std::endl;
      std::cout << *test_metric << " on test data = " << std::setprecision(4)
      << test_score << std::endl << std::endl;

      if (!scores_filename.empty()) {
        std::ofstream os;
        os << std::setprecision(std::numeric_limits<Score>::digits10);
        os.open(scores_filename, std::fstream::out);
        for (size_t i = 0; i < test_dataset->num_instances(); ++i)
          os << scores[i] << std::endl;
        os.close();
        std::cout << "# Scores written to file: " << scores_filename << std::endl;
      }
    }
  }

  algo->print_additional_stats();
}

}  // namespace metric
}  // namespace quickrank
