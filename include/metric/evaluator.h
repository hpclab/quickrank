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
  /// \param test_metric The metric measured on the test data.
  /// \param training_filename The training dataset.
  /// \param validation_filename The validation dataset.
  /// If empty, validation is not used.
  /// \param test_filename The test dataset.
  /// If empty, no performance is measured on the test set.
  /// \param output_filename Model output file.
  /// If empty, no output file is written.
  static void evaluate(learning::LTR_Algorithm* algo, ir::Metric* train_metric,
                       ir::Metric* test_metric,
                       const std::string training_filename,
                       const std::string validation_filename,
                       const std::string test_filename,
                       const std::string feature_filename,
                       const std::string output_filename);

};

}  // namespace metric
}  // namespace quickrank

#endif

