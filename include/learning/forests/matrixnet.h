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
#ifndef QUICKRANK_LEARNING_FORESTS_MATRIXNET_H_
#define QUICKRANK_LEARNING_FORESTS_MATRIXNET_H_

#include "types.h"
#include "learning/forests/lambdamart.h"
#include "learning/tree/ot.h"
#include "learning/tree/ensemble.h"

namespace quickrank {
namespace learning {
namespace forests {

class MatrixNet : public LambdaMart {
 public:
  /// Initializes a new MatrixNet instance with the given learning parameters.
  ///
  /// \param ntrees Maximum number of trees.
  /// \param shrinkage Learning rate.
  /// \param nthresholds Number of bins in discretization. 0 means no discretization.
  /// \param treedepth Maximum depth of each tree.
  /// \param minleafsupport Minimum number of instances in each leaf.
  /// \param esr Early stopping if no improvement after \esr iterations
  /// on the validation set.
  MatrixNet(unsigned int ntrees, float shrinkage, unsigned int nthresholds,
            unsigned int treedepth, unsigned int minleafsupport,
            unsigned int esr)
      : LambdaMart(ntrees, shrinkage, nthresholds, 1 << treedepth,
                   minleafsupport, esr),
        treedepth_(treedepth) {
  }

  MatrixNet(const boost::property_tree::ptree &info_ptree,
            const boost::property_tree::ptree &model_ptree);

  virtual ~MatrixNet() {
  }

  /// Returns the name of the ranker.
  virtual std::string name() const {
    return "MATRIXNET";
  }
  ;

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
  friend std::ostream& operator<<(std::ostream& os, const MatrixNet& a) {
    return a.put(os);
  }

  /// Prints the description of Algorithm, including its parameters.
  virtual std::ostream& put(std::ostream& os) const;

};

}  // namespace forests
}  // namespace learning
}  // namespace quickrank

#endif
