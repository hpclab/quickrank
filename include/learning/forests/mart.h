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
#ifndef QUICKRANK_LEARNING_FORESTS_MART_H_
#define QUICKRANK_LEARNING_FORESTS_MART_H_

#include "types.h"
#include "learning/ltr_algorithm.h"
#include "learning/tree/rt.h"
#include "learning/tree/ensemble.h"

namespace quickrank {
namespace learning {
namespace forests {

class Mart : public LTR_Algorithm {
 public:
  /// Initializes a new Mart instance with the given learning parameters.
  ///
  /// \param ntrees Maximum number of trees.
  /// \param shrinkage Learning rate.
  /// \param nthresholds Number of bins in discretization. 0 means no discretization.
  /// \param ntreeleaves Maximum number of leaves in each tree.
  /// \param minleafsupport Minimum number of instances in each leaf.
  /// \param valid_iterations Early stopping if no improvement after \esr iterations
  /// on the validation set.
  Mart(unsigned int ntrees, float shrinkage, unsigned int nthresholds,
       unsigned int ntreeleaves, unsigned int minleafsupport,
       unsigned int valid_iterations)
      : ntrees_(ntrees),
        shrinkage_(shrinkage),
        nthresholds_(nthresholds),
        nleaves_(ntreeleaves),
        minleafsupport_(minleafsupport),
        valid_iterations_(valid_iterations) {
  }

  /// Generates a LTR_Algorithm instance from a previously saved XML model.
  Mart(const boost::property_tree::ptree &info_ptree,
       const boost::property_tree::ptree &model_ptree);

  virtual ~Mart() {
  }

  /// Start the learning process.
  virtual void learn(std::shared_ptr<data::Dataset> training_dataset,
                     std::shared_ptr<data::Dataset> validation_dataset,
                     std::shared_ptr<metric::ir::Metric> training_metric,
                     unsigned int partial_save,
                     const std::string output_basename);

  /// Returns the score by the current ranker
  ///
  /// \param d Document to be scored.
  /// \param offset Offset to the next feature from \a d.
  virtual Score score_document(const Feature* d,
                               const unsigned int offset = 1) const {
    return ensemble_model_.score_instance(d, offset);
  }

  /// Print additional statistics.
  ///
  /// At the moment this include only number of comparisons for tree-based algorithms.
  virtual void print_additional_stats(void) const;

  /// Returns the name of the ranker.
  virtual std::string name() const {
    return NAME_;
  }

  static const std::string NAME_;

 protected:
  /// Makes sure the dataset in in vertical format.
  virtual void preprocess_dataset(std::shared_ptr<data::Dataset> dataset) const;

  /// Prepares private data structures before training takes place.
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

  /// Updates scores with the last learnt regression tree.
  ///
  /// \param dataset Dataset to be scored.
  /// \param scores Scores vector to be updated.
  /// \param tree Last regression tree leartn.
  virtual void update_modelscores(std::shared_ptr<data::Dataset> dataset,
                                  Score *scores, RegressionTree* tree);

  virtual std::ofstream& save_model_to_file(std::ofstream& os) const;

 protected:
  float **thresholds_ = NULL;
  unsigned int *thresholds_size_ = NULL;
  double *scores_on_training_ = NULL;  //[0..nentries-1]
  quickrank::Score* scores_on_validation_ = NULL;  //[0..nentries-1]
  unsigned int validation_bestmodel_ = 0;
  double *pseudoresponses_ = NULL;  //[0..nentries-1]
  Ensemble ensemble_model_;

  unsigned int ntrees_;  //>0
  double shrinkage_;  //>0.0f
  unsigned int nthresholds_;  //if ==0 then no. of thresholds is not limited
  unsigned int nleaves_;  //>0
  unsigned int minleafsupport_;  //>0
  unsigned int valid_iterations_;  //If no performance gain on validation data is observed in 'esr' rounds, stop the training process right away (if esr==0 feature is disabled).

  unsigned int **sortedsid_ = NULL;
  unsigned int sortedsize_ = 0;
  RTRootHistogram *hist_ = NULL;

 private:
  /// The output stream operator.
  friend std::ostream& operator<<(std::ostream& os, const Mart& a) {
    return a.put(os);
  }

  /// Prints the description of Algorithm, including its parameters.
  virtual std::ostream& put(std::ostream& os) const;

};

}  // namespace forests
}  // namespace learning
}  // namespace quickrank

#endif
