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
                              const std::string scores_filename) {
  if (test_metric and !test_filename.empty()) {

    // create reader: assum svml as ltr format
    quickrank::io::Svml reader;

    std::cout << "# Reading test dataset: " << test_filename << std::endl;

    std::shared_ptr<quickrank::data::Dataset> test_dataset = reader
        .read_horizontal(test_filename);
    std::cout << reader << *test_dataset;
    Score* test_scores = new Score[test_dataset
        ->num_instances()];
    algo->score_dataset(test_dataset, test_scores);
    quickrank::MetricScore test_score = test_metric->evaluate_dataset(
        test_dataset, test_scores);

    std::cout << std::endl;
    std::cout << *test_metric << " on test data = " << std::setprecision(4)
              << test_score << std::endl << std::endl;

    if (!scores_filename.empty()) {
      std::ofstream os;
      os << std::setprecision(std::numeric_limits<Score>::digits10);
      os.open(scores_filename, std::fstream::out);
      for (unsigned int i = 0; i < test_dataset->num_instances(); ++i)
        os << test_scores[i] << std::endl;
      os.close();
      std::cout << "# Scores written to file: " << scores_filename << std::endl;
    }

    delete[] test_scores;
  }
}

}  // namespace metric
}  // namespace quickrank
