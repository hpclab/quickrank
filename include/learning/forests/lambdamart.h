#ifndef QUICKRANK_LEARNING_FORESTS_LMART_H_
#define QUICKRANK_LEARNING_FORESTS_LMART_H_

#include "types.h"
#include "learning/forests/mart.h"
#include "learning/tree/rt.h"
#include "learning/tree/ensemble.h"

namespace quickrank {
namespace learning {
namespace forests {

class LambdaMart : public Mart {
 public:
  /// Initializes a new LambdaMart instance with the given learning parameters.
  ///
  /// \param ntrees Maximum number of trees.
  /// \param shrinkage Learning rate.
  /// \param nthresholds Number of bins in discretization. 0 means no discretization.
  /// \param ntreeleaves Maximum number of leaves in each tree.
  /// \param minleafsupport Minimum number of instances in each leaf.
  /// \param esr Early stopping if no improvement after \esr iterations
  /// on the validation set.
  LambdaMart(unsigned int ntrees, float shrinkage, unsigned int nthresholds,
             unsigned int ntreeleaves, unsigned int minleafsupport,
             unsigned int esr)
      : Mart(ntrees, shrinkage, nthresholds, ntreeleaves, minleafsupport, esr) {
  }

  /// Generates a LTR_Algorithm instance from a previously saved XML model.
  LambdaMart(const boost::property_tree::ptree &info_ptree,
             const boost::property_tree::ptree &model_ptree)
      : Mart(info_ptree, model_ptree);

  virtual ~LambdaMart() {
  }

  /// Returns the name of the ranker.
  virtual std::string name() const {
    return "LAMBDAMART";
  }

 protected:
  /// Prepares private data structurs befor training takes place.
  virtual void init(std::shared_ptr<data::Dataset> training_dataset,
                    std::shared_ptr<data::Dataset> validation_dataset);

  /// De-allocates private data structure after training has taken place.
  virtual void clear(std::shared_ptr<data::Dataset> training_dataset);

  /// Computes pseudo responses.
  ///
  /// \param training_dataset The training data.
  /// \param metric The metric to be optimized.
  virtual void compute_pseudoresponses(
      std::shared_ptr<data::Dataset> training_dataset,
      metric::ir::Metric* metric);

  /// Fits a regression tree on the gradient given by the pseudo residuals
  ///
  /// \param training_dataset The dataset used for training
  virtual std::unique_ptr<RegressionTree> fit_regressor_on_gradient(
      std::shared_ptr<data::Dataset> training_dataset);

  virtual std::ofstream& save_model_to_file(std::ofstream& os) const;

 private:
  /// The output stream operator.
  friend std::ostream& operator<<(std::ostream& os, const LambdaMart& a) {
    return a.put(os);
  }

  /// Prints the description of Algorithm, including its parameters.
  virtual std::ostream& put(std::ostream& os) const;

 protected:
  double* instance_weights_ = NULL;  //corresponds to datapoint.cache

};

}  // namespace forests
}  // namespace learning
}  // namespace quickrank

#endif
