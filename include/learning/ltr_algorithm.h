#ifndef QUICKRANK_LEARNING_LTR_ALGORITHM_H_
#define QUICKRANK_LEARNING_LTR_ALGORITHM_H_

#include "data/ltrdata.h"
#include "data/dataset.h"
#include "utils/qsort.h"

#include "metric/ir/metric.h"

namespace quickrank {
namespace learning {

class LTR_Algorithm {

 public:
  LTR_Algorithm() {
  }

  virtual ~LTR_Algorithm() {
  }

  // TODO: add load_model();

  virtual void score_dataset(quickrank::data::Dataset &dataset,
                             quickrank::Score* scores) const;

  // assumes vertical dataset
  // offset to next feature of the same instance
  virtual void score_query_results(
      std::shared_ptr<quickrank::data::QueryResults> results,
      quickrank::Score* scores, unsigned int offset) const;

  // assumes vertical dataset
  virtual quickrank::Score score_document(const quickrank::Feature* d,
                                          const unsigned int offset = 1) const;

  // TODO TO REMOVE
  virtual float eval_dp(float * const * const features,
                        unsigned int idx) const = 0;  //prediction value to store in a file

  virtual void learn(
      std::shared_ptr<quickrank::data::Dataset> training_dataset,
      std::shared_ptr<quickrank::data::Dataset> validation_dataset,
      quickrank::metric::ir::Metric*, unsigned int partial_save, const std::string output_basename) = 0;

  /// \deprecated This is something that will be removed when data structures will be consolidated.
  float compute_score(LTR_VerticalDataset *samples,
                      quickrank::metric::ir::Metric* scorer);

  /// Save the current model to the output_file.
  virtual void save(std::string output_basename, int = -1) const;

 private:

  /// The output stream operator.
  friend std::ostream& operator<<(std::ostream& os, const LTR_Algorithm& a) {
    return a.put(os);
  }

  /// Prints the description of Algorithm, including its parameters
  virtual std::ostream& put(std::ostream& os) const = 0;

  /// Save the current model in the given output file stream.
  virtual std::ofstream& save_model_to_file(std::ofstream& of) const = 0;
};

}  // namespace learning
}  // namespace quickrank

#endif
