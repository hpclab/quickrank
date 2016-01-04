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

#include "metric/evaluator.h"
#include "io/svml.h"

namespace quickrank {
namespace metric {

Evaluator::Evaluator() {
}

Evaluator::~Evaluator() {
}

void Evaluator::training_phase(std::shared_ptr<learning::LTR_Algorithm> algo,
                               std::shared_ptr<ir::Metric> train_metric,
                               const std::string training_filename,
                               const std::string validation_filename,
                               const std::string feature_filename,
                               const std::string output_filename,
                               const unsigned int npartialsave) {

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

void Evaluator::testing_phase(std::shared_ptr<learning::LTR_Algorithm> algo,
                              std::shared_ptr<ir::Metric> test_metric,
                              const std::string test_filename,
                              const std::string scores_filename,
                              const bool verbose) {
  if (test_metric and !test_filename.empty()) {

    // create reader: assume svml as ltr format
    quickrank::io::Svml svml;

    std::cout << "# Reading test dataset: " << test_filename << std::endl;

    std::shared_ptr<data::Dataset> test_dataset =
        svml.read_horizontal(test_filename);
    std::cout << svml << *test_dataset << std::endl;

    std::vector<Score> scores(test_dataset->num_instances());
    if (verbose) {
      unsigned int idx_query_scores = 0;
      std::shared_ptr<data::Dataset> datasetPartScores = nullptr;

      for (unsigned int q = 0; q < test_dataset->num_queries(); q++) {
        std::shared_ptr<data::QueryResults> results =
            test_dataset->getQueryResults(q);
        if (test_dataset->format() == data::Dataset::VERT) {
          std::cerr << "# ## ERROR!! Dataset should be in horiz format" <<
              std::endl;
          return;
        }
        // score_query_results(r, scores, 1, test_dataset->num_features());
        const Feature* features = results->features();
        const Label* labels = results->labels();
        for (unsigned int i = 0; i < results->num_results(); i++) {
          std::shared_ptr<std::vector<Score>> detailed_scores =
              algo->detailed_scores_document(features, 1);

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
        for (unsigned int i = 0; i < test_dataset->num_instances(); ++i)
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
