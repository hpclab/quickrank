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
      : Mart(info_ptree, model_ptree) {
  }

  virtual ~LambdaMart() {
  }

  /// Returns the name of the ranker.
  virtual std::string name() const {
    return NAME_;
  }

  static const std::string NAME_;

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

 protected:
  double* instance_weights_ = NULL;  //corresponds to datapoint.cache

};

}  // namespace forests
}  // namespace learning
}  // namespace quickrank

#endif
