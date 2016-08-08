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
#pragma once

#include <memory>

#include "metric/ir/metric.h"
#include "learning/ltr_algorithm.h"
#include "optimization/optimization.h"

#include "io/generate_vpred.h"
#include "io/generate_conditional_operators.h"
#include "io/generate_oblivious.h"

#include "paramsmap/paramsmap.h"

namespace quickrank {
namespace driver {

/**
 * This class implements the main logic of the quickrank application.
 */
class Driver {
 public:
  Driver();
  virtual ~Driver();

  /// Implements the main logic of the quickrank application, detecting
  /// the metrics to adopt and the phases to execute (train/validation/test).
  /// Returns the exit code of the application
  ///
  /// \param, vm The Variable mapping of CLI options (boost object)
  static int run(ParamsMap &pmap);

  static std::shared_ptr<data::Dataset> extract_partial_scores(
      std::shared_ptr<learning::LTR_Algorithm> algo,
      std::shared_ptr<data::Dataset> dataset,
      bool ignore_weights = false);

 private:
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
  static void training_phase(
      std::shared_ptr<learning::LTR_Algorithm> algo,
      std::shared_ptr<metric::ir::Metric> train_metric,
      std::shared_ptr<quickrank::data::Dataset> training_dataset,
      std::shared_ptr<quickrank::data::Dataset> validation_dataset,
      const std::string output_filename,
      const size_t npartialsave);

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
  static void optimization_phase(
      std::shared_ptr<quickrank::optimization::Optimization> opt_algorithm,
      std::shared_ptr<learning::LTR_Algorithm> ranking_algo,
      std::shared_ptr<metric::ir::Metric> train_metric,
      std::shared_ptr<quickrank::data::Dataset> training_dataset,
      std::shared_ptr<quickrank::data::Dataset> validation_dataset,
      std::string training_partial_filename,
      std::string validation_partial_filename,
      const std::string output_filename,
      const std::string opt_algo_model_filename,
      const size_t npartialsave);

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
  static void testing_phase(
      std::shared_ptr<learning::LTR_Algorithm> algo,
      std::shared_ptr<metric::ir::Metric> test_metric,
      std::shared_ptr<quickrank::data::Dataset> test_dataset,
      const std::string scores_filename,
      const bool detailed_testing);

  static std::shared_ptr<quickrank::data::Dataset> load_dataset(
      const std::string dataset_filename,
      const std::string dataset_label);
};

}  // namespace driver
}  // namespace quickrank

