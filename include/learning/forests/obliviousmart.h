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

#include "types.h"
#include "learning/forests/mart.h"
#include "learning/tree/ot.h"
#include "learning/tree/ensemble.h"

namespace quickrank {
namespace learning {
namespace forests {

class ObliviousMart: public Mart {
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
  ObliviousMart(size_t ntrees, double shrinkage, size_t nthresholds,
                size_t treedepth, size_t minleafsupport,
                size_t esr)
      : Mart(ntrees, shrinkage, nthresholds, 1 << treedepth, minleafsupport,
             esr),
        treedepth_(treedepth) {
  }

  ObliviousMart(const pugi::xml_document &model);

  virtual ~ObliviousMart() {
  }

  /// Returns the name of the ranker.
  virtual std::string name() const {
    return NAME_;
  }

  virtual bool import_model_state(LTR_Algorithm &other);

  static const std::string NAME_;

 protected:
  /// Fits a regression tree on the gradient given by the pseudo residuals
  ///
  /// \param training_dataset The dataset used for training
  virtual std::unique_ptr<RegressionTree> fit_regressor_on_gradient(
      std::shared_ptr<data::VerticalDataset> training_dataset);

  virtual pugi::xml_document *get_xml_model() const;

  size_t treedepth_;  //>0

 private:
  /// The output stream operator.
  friend std::ostream &operator<<(std::ostream &os, const ObliviousMart &a) {
    return a.put(os);
  }

  /// Prints the description of Algorithm, including its parameters.
  virtual std::ostream &put(std::ostream &os) const;

};

}  // namespace forests
}  // namespace learning
}  // namespace quickrank
