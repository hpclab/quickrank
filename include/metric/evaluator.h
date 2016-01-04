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
#ifndef QUICKRANK_METRIC_EVALUATOR_H_
#define QUICKRANK_METRIC_EVALUATOR_H_

#include "metric/ir/metric.h"
#include "learning/ltr_algorithm.h"

namespace quickrank {
namespace metric {

/**
 * This class implements some utility functions to train and test L-t-R models.
 */
class Evaluator : private boost::noncopyable {
 public:
  Evaluator();
  virtual ~Evaluator();

  /// Runs train/validation of \a algo by optimizing \a train_metric
  /// and then measures \a test_metric on the test data.
  ///
  /// \param algo The L-T-R algorithm to be tested.
  /// \param train_metric The metric optimized during training.
  /// \param training_filename The training dataset.
  /// \param validation_filename The validation dataset.
  /// If empty, validation is not used.
  /// \param output_filename Model output file.
  /// If empty, no output file is written.
  /// \param npartialsave Allows to save a partial model every given number of iterations.
  static void training_phase(std::shared_ptr<learning::LTR_Algorithm> algo,
                       std::shared_ptr<ir::Metric> train_metric,
                       const std::string training_filename,
                       const std::string validation_filename,
                       const std::string feature_filename,
                       const std::string model_filename,
                       const unsigned int npartialsave);

  /// Runs the learned or loaded model on the test data
  /// and then measures \a test_metric on the test data.
  ///
  /// \param algo The L-T-R algorithm to be tested.
  /// \param test_metric The metric measured on the test data.
  /// \param test_filename The test dataset.
  /// If empty, no performance is measured on the test set.
  /// \param scores_filename The output scores file.
  /// If set save the scores computed for the test set.
  /// \param verbose If True saves an SVML-like file with the score of each ranker in the ensemble.
  /// NB. Works only for ensembles.
  static void testing_phase(std::shared_ptr<learning::LTR_Algorithm> algo,
                       std::shared_ptr<ir::Metric> test_metric,
                       const std::string test_filename,
                       const std::string scores_filename,
                       const bool verbose);
};

}  // namespace metric
}  // namespace quickrank

#endif

