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
#ifndef QUICKRANK_LEARNING_FORESTS_OBLIVIOUSLAMBDAMART_H_
#define QUICKRANK_LEARNING_FORESTS_OBLIVIOUSLAMBDAMART_H_

#include "types.h"
#include "learning/forests/lambdamart.h"
#include "learning/tree/ot.h"
#include "learning/tree/ensemble.h"

namespace quickrank {
namespace learning {
namespace forests {

class ObliviousLambdaMart : public LambdaMart {
 public:
  /// Initializes a new ObliviousLambdaMart instance with the given learning parameters.
  ///
  /// \param ntrees Maximum number of trees.
  /// \param shrinkage Learning rate.
  /// \param nthresholds Number of bins in discretization. 0 means no discretization.
  /// \param treedepth Maximum depth of each tree.
  /// \param minleafsupport Minimum number of instances in each leaf.
  /// \param esr Early stopping if no improvement after \esr iterations
  /// on the validation set.
  ObliviousLambdaMart(unsigned int ntrees, float shrinkage, unsigned int nthresholds,
            unsigned int treedepth, unsigned int minleafsupport,
            unsigned int esr)
      : LambdaMart(ntrees, shrinkage, nthresholds, 1 << treedepth,
                   minleafsupport, esr),
        treedepth_(treedepth) {
  }

  ObliviousLambdaMart(const boost::property_tree::ptree &info_ptree,
            const boost::property_tree::ptree &model_ptree);

  virtual ~ObliviousLambdaMart() {
  }

  /// Returns the name of the ranker.
  virtual std::string name() const {
    return NAME_;
  }

  static const std::string NAME_;

 protected:
  /// Fits a regression tree on the gradient given by the pseudo residuals
  ///
  /// \param training_dataset The dataset used for training
  virtual std::unique_ptr<RegressionTree> fit_regressor_on_gradient(
      std::shared_ptr<data::Dataset> training_dataset);

  virtual std::ofstream& save_model_to_file(std::ofstream& os) const;

  unsigned int treedepth_;  //>0

 private:
  /// The output stream operator.
  friend std::ostream& operator<<(std::ostream& os, const ObliviousLambdaMart& a) {
    return a.put(os);
  }

  /// Prints the description of Algorithm, including its parameters.
  virtual std::ostream& put(std::ostream& os) const;

};

}  // namespace forests
}  // namespace learning
}  // namespace quickrank

#endif
